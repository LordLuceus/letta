"""Unit tests for MoonshotClient.

Covers Moonshot-specific request shaping:
  - K2.x parameter restrictions (temperature, top_p, frequency/presence penalty, n stripped)
  - reasoning_content backfill on assistant tool-call messages so multi-turn
    tool calling works with thinking models (kimi-k2.5, kimi-k2.6, kimi-k2-thinking).
    Without this, Moonshot returns 400 "thinking is enabled but reasoning_content
    is missing in assistant tool call message at index N".
  - Orphan tool_call sanitization: drop assistant tool_calls whose tool_call_id
    has no matching tool response message (and drop tool messages whose
    tool_call_id has no matching assistant call). Without this, compaction or
    history mutation can leave dangling references that cause Moonshot to 400
    with "an assistant message with 'tool_calls' must be followed by tool
    messages responding to each 'tool_call_id'".
"""

from unittest.mock import MagicMock, patch

import pytest

from letta.llm_api.moonshot_client import MoonshotClient
from letta.schemas.enums import AgentType
from letta.schemas.llm_config import LLMConfig

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"


def _make_assistant_tool_call_msg(content=None, reasoning_content=None):
    """Build a fake assistant message dict with a tool_call."""
    msg = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "send_message", "arguments": '{"text": "hi"}'},
            }
        ],
    }
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    return msg


def _make_user_msg(text="hello"):
    return {"role": "user", "content": text}


def _make_tool_result_msg(text="ok"):
    return {"role": "tool", "tool_call_id": "call_123", "content": text}


class TestMoonshotClientK2ParamStripping:
    """K2.x models reject non-default sampling params; we strip them."""

    def setup_method(self):
        self.client = MoonshotClient(put_inner_thoughts_first=True)
        self.llm_config = LLMConfig(
            model="kimi-k2.6",
            model_endpoint_type="moonshot",
            model_endpoint=MOONSHOT_BASE_URL,
            context_window=200000,
        )

    def test_k2_strips_sampling_params(self):
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "temperature": 0.5,
                "top_p": 0.8,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "n": 2,
                "messages": [_make_user_msg()],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.llm_config, [],
            )

            assert "temperature" not in result
            assert "top_p" not in result
            assert "frequency_penalty" not in result
            assert "presence_penalty" not in result
            assert "n" not in result

    def test_non_k2_keeps_sampling_params(self):
        config = LLMConfig(
            model="moonshot-v1-128k",
            model_endpoint_type="moonshot",
            model_endpoint=MOONSHOT_BASE_URL,
            context_window=128000,
        )
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "moonshot-v1-128k",
                "temperature": 0.5,
                "top_p": 0.8,
                "messages": [_make_user_msg()],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], config, [],
            )

            assert result["temperature"] == 0.5
            assert result["top_p"] == 0.8


class TestMoonshotClientReasoningContentBackfill:
    """Moonshot K2 thinking models require reasoning_content on every assistant tool-call message.

    Old conversation history won't have this field on tool-call messages from before
    thinking was enabled, so we backfill a placeholder for those messages to satisfy
    the API validation.
    """

    def setup_method(self):
        self.client = MoonshotClient(put_inner_thoughts_first=True)
        self.k2_config = LLMConfig(
            model="kimi-k2.6",
            model_endpoint_type="moonshot",
            model_endpoint=MOONSHOT_BASE_URL,
            context_window=200000,
        )
        self.legacy_config = LLMConfig(
            model="moonshot-v1-128k",
            model_endpoint_type="moonshot",
            model_endpoint=MOONSHOT_BASE_URL,
            context_window=128000,
        )

    def test_k2_backfills_missing_reasoning_content_on_tool_call_messages(self):
        """Old assistant tool-call messages without reasoning_content get a placeholder."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg("first turn"),
                    _make_assistant_tool_call_msg(content="thinking..."),  # no reasoning_content
                    _make_tool_result_msg(),
                    _make_user_msg("second turn"),
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            assistant_msg = result["messages"][1]
            assert "reasoning_content" in assistant_msg
            # Placeholder must be non-empty (Moonshot rejects None/missing)
            assert assistant_msg["reasoning_content"]

    def test_k2_preserves_existing_reasoning_content(self):
        """If reasoning_content is already set, don't overwrite it."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg(),
                    _make_assistant_tool_call_msg(
                        content="answer",
                        reasoning_content="actual reasoning text",
                    ),
                    _make_tool_result_msg(),
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            assistant_msg = result["messages"][1]
            assert assistant_msg["reasoning_content"] == "actual reasoning text"

    def test_k2_does_not_add_reasoning_content_to_non_tool_call_assistant_messages(self):
        """Plain assistant text replies don't need reasoning_content backfilled."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg(),
                    {"role": "assistant", "content": "plain reply, no tools"},
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            assistant_msg = result["messages"][1]
            assert "reasoning_content" not in assistant_msg

    def test_k2_does_not_touch_user_or_tool_messages(self):
        """User and tool-result messages should never get reasoning_content."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg("hi"),
                    _make_assistant_tool_call_msg(),
                    _make_tool_result_msg("result"),
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            assert "reasoning_content" not in result["messages"][0]
            assert "reasoning_content" not in result["messages"][2]

    def test_legacy_v1_models_do_not_get_backfill(self):
        """moonshot-v1-* models don't have thinking; backfill should be skipped."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "moonshot-v1-128k",
                "messages": [
                    _make_user_msg(),
                    _make_assistant_tool_call_msg(),
                    _make_tool_result_msg(),
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.legacy_config, [],
            )

            assistant_msg = result["messages"][1]
            assert "reasoning_content" not in assistant_msg

    def test_k2_handles_multiple_tool_call_messages(self):
        """Backfill applies across all tool-call messages in history, not just one."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg("q1"),
                    _make_assistant_tool_call_msg(content="t1"),
                    _make_tool_result_msg("r1"),
                    _make_user_msg("q2"),
                    _make_assistant_tool_call_msg(content="t2"),
                    _make_tool_result_msg("r2"),
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            assert result["messages"][1].get("reasoning_content")
            assert result["messages"][4].get("reasoning_content")


def _make_assistant_with_tool_calls(call_ids, content=None):
    """Build an assistant message with multiple tool_calls."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "Read", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    }


def _make_tool_msg(call_id, text="ok"):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


class TestMoonshotClientOrphanToolCallSanitization:
    """Drop orphan tool_calls / tool messages so compaction-induced gaps don't 400.

    Moonshot (and OpenAI) reject any request where an assistant tool_call has
    no matching tool response in the same message list. Compaction or
    interrupted runs can leave such orphans behind. We sanitize the payload
    defensively before hitting the wire.
    """

    def setup_method(self):
        self.client = MoonshotClient(put_inner_thoughts_first=True)
        self.k2_config = LLMConfig(
            model="kimi-k2.6",
            model_endpoint_type="moonshot",
            model_endpoint=MOONSHOT_BASE_URL,
            context_window=200000,
        )
        self.legacy_config = LLMConfig(
            model="moonshot-v1-128k",
            model_endpoint_type="moonshot",
            model_endpoint=MOONSHOT_BASE_URL,
            context_window=128000,
        )

    def test_healthy_messages_pass_through_unchanged(self):
        """No orphans → no changes to message structure."""
        messages = [
            _make_user_msg("hi"),
            _make_assistant_with_tool_calls(["call_a"], content="thinking"),
            _make_tool_msg("call_a", "result"),
            _make_user_msg("next"),
        ]
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {"model": "kimi-k2.6", "messages": messages}
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            assistant = result["messages"][1]
            assert assistant.get("tool_calls"), "tool_calls should be preserved"
            assert len(assistant["tool_calls"]) == 1
            assert assistant["tool_calls"][0]["id"] == "call_a"
            # tool message should still be there
            assert result["messages"][2]["tool_call_id"] == "call_a"

    def test_partial_orphan_drops_offending_call_only(self):
        """Assistant with two tool_calls, one orphan: keep the responded one, drop orphan."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg(),
                    _make_assistant_with_tool_calls(["call_kept", "Read:206"], content="x"),
                    _make_tool_msg("call_kept"),
                    # No tool message for "Read:206" -> orphan
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            assistant = result["messages"][1]
            ids = [tc["id"] for tc in assistant.get("tool_calls", [])]
            assert "call_kept" in ids
            assert "Read:206" not in ids

    def test_full_orphan_removes_tool_calls_field_entirely(self):
        """Assistant whose tool_calls are ALL orphans loses the tool_calls field.

        Content must be a non-None string so the message remains valid.
        """
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg(),
                    _make_assistant_with_tool_calls(["Read:206", "Edit:42"], content=None),
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            assistant = result["messages"][1]
            assert not assistant.get("tool_calls"), "all-orphan tool_calls should be removed"
            # content must be coercible — None is invalid for assistant w/o tool_calls
            assert assistant.get("content") is not None

    def test_full_orphan_preserves_existing_content(self):
        """If the assistant already has text content, keep it as-is."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg(),
                    _make_assistant_with_tool_calls(["Read:206"], content="my reply"),
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            assistant = result["messages"][1]
            assert not assistant.get("tool_calls")
            assert assistant["content"] == "my reply"

    def test_orphan_tool_message_dropped(self):
        """Tool message with no matching assistant tool_call is dropped."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg(),
                    _make_assistant_with_tool_calls(["call_a"], content="x"),
                    _make_tool_msg("call_a", "ok"),
                    _make_tool_msg("ghost_id", "from nowhere"),  # orphan
                    _make_user_msg("next"),
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            tool_call_ids_in_result = [
                m.get("tool_call_id")
                for m in result["messages"]
                if m.get("role") == "tool"
            ]
            assert "call_a" in tool_call_ids_in_result
            assert "ghost_id" not in tool_call_ids_in_result

    def test_sanitization_runs_for_legacy_v1_models(self):
        """Orphan sanitization is provider-wide, not gated on K2."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "moonshot-v1-128k",
                "messages": [
                    _make_user_msg(),
                    _make_assistant_with_tool_calls(["Read:99"], content="x"),
                    # No matching tool message
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.legacy_config, [],
            )

            assistant = result["messages"][1]
            assert not assistant.get("tool_calls")

    def test_sanitization_preserves_message_order(self):
        """Order of remaining messages must be preserved (no reordering)."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg("first"),
                    _make_assistant_with_tool_calls(["a"], content="t1"),
                    _make_tool_msg("a", "r1"),
                    _make_user_msg("second"),
                    _make_assistant_with_tool_calls(["b", "Read:206"], content="t2"),
                    _make_tool_msg("b", "r2"),
                    _make_user_msg("third"),
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            roles = [m["role"] for m in result["messages"]]
            assert roles == ["user", "assistant", "tool", "user", "assistant", "tool", "user"]
            # And the orphan was dropped from the second assistant
            ids = [tc["id"] for tc in result["messages"][4]["tool_calls"]]
            assert "b" in ids and "Read:206" not in ids

    def test_real_reproduction_read_206(self):
        """Direct repro of the user's failing payload."""
        with patch.object(MoonshotClient.__bases__[0], "build_request_data") as mock_super:
            mock_super.return_value = {
                "model": "kimi-k2.6",
                "messages": [
                    _make_user_msg("do work"),
                    _make_assistant_with_tool_calls(["Read:206"], content="Reading file"),
                    # No tool response message — this is what triggers the 400
                    _make_user_msg("continue"),
                ],
            }
            result = self.client.build_request_data(
                AgentType.letta_v1_agent, [], self.k2_config, [],
            )

            # No assistant in the result should have an orphan tool_call
            for m in result["messages"]:
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        # Every kept tool_call must have a matching tool message
                        matching = [
                            tm for tm in result["messages"]
                            if tm.get("role") == "tool" and tm.get("tool_call_id") == tc["id"]
                        ]
                        assert matching, f"Kept tool_call {tc['id']} has no tool response"
