"""
Tests for the ConversationIdOrDefault validator that accepts agent-* IDs
and 'default' for agent-direct messaging (compatibility with Letta Code >= 0.16.13).

Upstream uses ConversationIdOrDefault (separate from plain ConversationId)
for endpoints that support agent-direct mode.
"""

import re
from typing import get_args

import pytest
from annotated_types import MaxLen, MinLen

from letta.schemas.enums import PrimitiveType
from letta.validators import PRIMITIVE_ID_PATTERNS, ConversationIdOrDefault


def _get_conversation_id_or_default_path():
    """Get the Path validator from the ConversationIdOrDefault type alias."""
    # ConversationIdOrDefault is Annotated[str, Path(...)], so the Path is in __metadata__
    args = get_args(ConversationIdOrDefault)
    # args[0] is str, args[1] is the Path validator
    return args[1]


def _get_conversation_id_or_default_pattern() -> re.Pattern:
    """Extract the compiled regex from ConversationIdOrDefault path validator metadata."""
    path_obj = _get_conversation_id_or_default_path()
    for m in path_obj.metadata:
        if hasattr(m, "pattern"):
            return re.compile(m.pattern)
    raise RuntimeError("Could not find pattern in ConversationIdOrDefault metadata")


class TestConversationIdValidatorPattern:
    """Test that the ConversationId regex accepts the right formats."""

    @pytest.fixture
    def pattern(self):
        return _get_conversation_id_or_default_pattern()

    # --- Valid IDs ---

    def test_accepts_default(self, pattern):
        assert pattern.match("default")

    def test_accepts_conv_uuid(self, pattern):
        assert pattern.match("conv-123e4567-e89b-42d3-8456-426614174000")

    def test_accepts_agent_uuid(self, pattern):
        """Letta Code >= 0.16.13 sends agent-* IDs to conversations endpoint."""
        assert pattern.match("agent-123e4567-e89b-42d3-8456-426614174000")

    def test_accepts_agent_uuid_various(self, pattern):
        """Test multiple valid agent UUIDs."""
        valid_agent_ids = [
            "agent-00000000-0000-4000-8000-000000000000",
            "agent-ffffffff-ffff-4fff-bfff-ffffffffffff",
            "agent-a1b2c3d4-e5f6-4789-abcd-ef0123456789",
        ]
        for agent_id in valid_agent_ids:
            assert pattern.match(agent_id), f"Should accept {agent_id}"

    # --- Invalid IDs ---

    def test_rejects_empty(self, pattern):
        assert not pattern.match("")

    def test_rejects_random_string(self, pattern):
        assert not pattern.match("foobar")

    def test_rejects_wrong_prefix(self, pattern):
        assert not pattern.match("invalid-123e4567-e89b-42d3-8456-426614174000")

    def test_rejects_tool_prefix(self, pattern):
        assert not pattern.match("tool-123e4567-e89b-42d3-8456-426614174000")

    def test_rejects_agent_without_uuid(self, pattern):
        assert not pattern.match("agent-not-a-uuid")

    def test_rejects_conv_without_uuid(self, pattern):
        assert not pattern.match("conv-not-a-uuid")

    def test_rejects_agent_prefix_only(self, pattern):
        assert not pattern.match("agent-")

    def test_rejects_conv_prefix_only(self, pattern):
        assert not pattern.match("conv-")


class TestConversationIdValidatorLengths:
    """Test that min/max length constraints work."""

    @staticmethod
    def _get_length_constraint(path_obj, cls):
        """Extract a MinLen or MaxLen from Path metadata."""
        for m in path_obj.metadata:
            if isinstance(m, cls):
                return m.max_length if cls is MaxLen else m.min_length
        raise RuntimeError(f"Could not find {cls.__name__} in metadata")

    def test_max_length_accommodates_agent_ids(self):
        """agent-<uuid> is 42 chars, conv-<uuid> is 41 chars. Max must be >= 42."""
        path_obj = _get_conversation_id_or_default_path()
        max_len = self._get_length_constraint(path_obj, MaxLen)
        agent_id = "agent-123e4567-e89b-42d3-8456-426614174000"
        assert len(agent_id) == 42
        assert max_len >= 42

    def test_min_length_allows_default(self):
        """'default' is 7 chars. Min must be <= 7."""
        path_obj = _get_conversation_id_or_default_path()
        min_len = self._get_length_constraint(path_obj, MinLen)
        assert min_len <= len("default")


class TestIsAgentIdHelper:
    """Test the _is_agent_id routing helper from conversations.py."""

    @staticmethod
    def _is_agent_id(conversation_id: str) -> bool:
        """Replicate the helper from conversations.py for isolated testing."""
        return conversation_id.startswith("agent-")

    def test_agent_id_detected(self):
        assert self._is_agent_id("agent-123e4567-e89b-42d3-8456-426614174000")

    def test_conv_id_not_detected(self):
        assert not self._is_agent_id("conv-123e4567-e89b-42d3-8456-426614174000")

    def test_default_not_detected(self):
        assert not self._is_agent_id("default")

    def test_empty_not_detected(self):
        assert not self._is_agent_id("")
