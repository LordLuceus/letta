import os
from typing import List, Optional

from openai import AsyncOpenAI, AsyncStream, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from letta.helpers.json_helpers import sanitize_unicode_surrogates
from letta.llm_api.openai_client import OpenAIClient
from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.schemas.enums import AgentType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message as PydanticMessage
from letta.settings import model_settings

logger = get_logger(__name__)


def _is_k2_model(model: str) -> bool:
    """Check if the model is a Kimi K2.x model (which has parameter restrictions)."""
    return model.startswith("kimi-k2")


# Placeholder reasoning_content used to backfill historical assistant tool-call
# messages so the Moonshot K2 thinking API accepts the request. A single space
# is the minimal non-empty string the API will accept.
_REASONING_CONTENT_PLACEHOLDER = " "


def _sanitize_orphan_tool_calls(messages: list) -> list:
    """Drop assistant tool_calls without matching tool responses, and tool messages
    without matching assistant tool_calls.

    Moonshot (and OpenAI proper) reject any request where an assistant message's
    ``tool_calls`` array contains an id that no subsequent ``tool`` message
    responds to. Compaction, mid-run interruptions, or storage drift can leave
    such orphans. This function removes them defensively so the request shape
    is always valid.

    Order of remaining messages is preserved. Mutates message dicts in-place
    where it can; returns a new list with any fully-orphan tool messages removed.

    Returns the (possibly filtered) list of message dicts.
    """
    if not messages:
        return messages

    # Pass 1: collect every tool_call_id that has a matching tool response.
    responded_ids: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if isinstance(tcid, str):
            responded_ids.add(tcid)

    dropped_calls = 0
    dropped_tool_msgs = 0

    # Pass 2: for each assistant message, drop tool_calls whose id has no response.
    # Track which tool_call_ids survive so we can drop orphan tool messages too.
    surviving_call_ids: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue
        kept = []
        for tc in tool_calls:
            tcid = tc.get("id") if isinstance(tc, dict) else None
            if isinstance(tcid, str) and tcid in responded_ids:
                kept.append(tc)
                surviving_call_ids.add(tcid)
            else:
                dropped_calls += 1
        if len(kept) != len(tool_calls):
            if kept:
                msg["tool_calls"] = kept
            else:
                # All tool_calls were orphans. Drop the field entirely and ensure
                # content is a non-None string so the assistant message stays valid.
                msg.pop("tool_calls", None)
                if msg.get("content") is None:
                    msg["content"] = ""

    # Pass 3: drop tool messages that don't correspond to any surviving tool_call.
    cleaned: list = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tcid = msg.get("tool_call_id")
            if not isinstance(tcid, str) or tcid not in surviving_call_ids:
                dropped_tool_msgs += 1
                continue
        cleaned.append(msg)

    if dropped_calls or dropped_tool_msgs:
        logger.warning(
            "[Moonshot] Sanitized orphan tool references before request: "
            "dropped_tool_calls=%d, dropped_tool_messages=%d",
            dropped_calls,
            dropped_tool_msgs,
        )

    return cleaned


class MoonshotClient(OpenAIClient):
    """
    LLM client for Moonshot AI (Kimi) models.

    Kimi K2.x models have strict parameter requirements:
    - temperature must be 1.0 (thinking) or 0.6 (non-thinking); other values error
    - top_p must be 0.95; other values error
    - frequency_penalty must be 0.0; other values error
    - presence_penalty must be 0.0; other values error
    - n must be 1; other values error

    We strip these params and let the API use its own defaults.

    Kimi K2.x thinking models additionally require ``reasoning_content`` on every
    assistant message that contains ``tool_calls``. Old conversation history (or
    runs that started before thinking was enabled) won't have this field, which
    causes the API to return 400 "thinking is enabled but reasoning_content is
    missing in assistant tool call message at index N". We backfill a placeholder
    on any tool-call assistant message that lacks it so multi-turn tool calling
    works against thinking-capable K2 models.
    """

    def requires_auto_tool_choice(self, llm_config: LLMConfig) -> bool:
        # K2.x in thinking mode only supports "auto" or "none" for tool_choice
        return True

    def supports_structured_output(self, llm_config: LLMConfig) -> bool:
        return False

    @trace_method
    def build_request_data(
        self,
        agent_type: AgentType,
        messages: List[PydanticMessage],
        llm_config: LLMConfig,
        tools: Optional[List[dict]] = None,
        force_tool_call: Optional[str] = None,
        requires_subsequent_tool_call: bool = False,
        tool_return_truncation_chars: Optional[int] = None,
        system: Optional[str] = None,
    ) -> dict:
        data = super().build_request_data(
            agent_type,
            messages,
            llm_config,
            tools,
            force_tool_call,
            requires_subsequent_tool_call,
            tool_return_truncation_chars,
            system,
        )

        model = data.get("model", "")

        if _is_k2_model(model):
            # K2.x models require specific fixed values for these params;
            # any non-default value returns an error. Strip them so the API
            # uses its own defaults.
            data.pop("temperature", None)
            data.pop("top_p", None)
            data.pop("frequency_penalty", None)
            data.pop("presence_penalty", None)
            data.pop("n", None)

        # Drop orphan tool_calls / tool messages so the request can't 400 with
        # "an assistant message with 'tool_calls' must be followed by tool
        # messages responding to each 'tool_call_id'". Run for ALL Moonshot
        # models (K2 and v1) since this is a Moonshot-wide validation.
        if "messages" in data:
            data["messages"] = _sanitize_orphan_tool_calls(data["messages"])

        if _is_k2_model(model):
            # Backfill reasoning_content on assistant tool-call messages that
            # lack it. K2 thinking models reject the request otherwise.
            # Run AFTER orphan sanitization so we only backfill on surviving
            # tool_call assistants.
            for msg in data.get("messages", []):
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "assistant":
                    continue
                if not msg.get("tool_calls"):
                    continue
                if not msg.get("reasoning_content"):
                    msg["reasoning_content"] = _REASONING_CONTENT_PLACEHOLDER

        return data

    @trace_method
    def request(self, request_data: dict, llm_config: LLMConfig) -> dict:
        request_data = sanitize_unicode_surrogates(request_data)

        api_key = model_settings.moonshot_api_key or os.environ.get("MOONSHOT_API_KEY")
        client = OpenAI(api_key=api_key, base_url=llm_config.model_endpoint)

        response: ChatCompletion = client.chat.completions.create(**request_data)
        return response.model_dump()

    @trace_method
    async def request_async(self, request_data: dict, llm_config: LLMConfig) -> dict:
        request_data = sanitize_unicode_surrogates(request_data)

        api_key = model_settings.moonshot_api_key or os.environ.get("MOONSHOT_API_KEY")
        client = AsyncOpenAI(api_key=api_key, base_url=llm_config.model_endpoint)

        response: ChatCompletion = await client.chat.completions.create(**request_data)
        return response.model_dump()

    @trace_method
    async def stream_async(self, request_data: dict, llm_config: LLMConfig) -> AsyncStream[ChatCompletionChunk]:
        request_data = sanitize_unicode_surrogates(request_data)

        api_key = model_settings.moonshot_api_key or os.environ.get("MOONSHOT_API_KEY")
        client = AsyncOpenAI(api_key=api_key, base_url=llm_config.model_endpoint)
        response_stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
            **request_data, stream=True, stream_options={"include_usage": True}
        )
        return response_stream
