"""Unit tests for MoonshotClient.

Covers Moonshot-specific request shaping:
  - K2.x parameter restrictions (temperature, top_p, frequency/presence penalty, n stripped)
  - reasoning_content backfill on assistant tool-call messages so multi-turn
    tool calling works with thinking models (kimi-k2.5, kimi-k2.6, kimi-k2-thinking).
    Without this, Moonshot returns 400 "thinking is enabled but reasoning_content
    is missing in assistant tool call message at index N".
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
