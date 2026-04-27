import os
from typing import List, Optional

from openai import AsyncOpenAI, AsyncStream, OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from letta.helpers.json_helpers import sanitize_unicode_surrogates
from letta.llm_api.openai_client import OpenAIClient
from letta.otel.tracing import trace_method
from letta.schemas.enums import AgentType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message as PydanticMessage
from letta.settings import model_settings


def _is_k2_model(model: str) -> bool:
    """Check if the model is a Kimi K2.x model (which has parameter restrictions)."""
    return model.startswith("kimi-k2")


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
