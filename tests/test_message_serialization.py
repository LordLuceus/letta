from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from letta.llm_api.openai_client import fill_image_content_in_responses_input
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent
from letta.schemas.message import Message


def _user_message_with_image_first(text: str) -> Message:
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    return Message(role=MessageRole.user, content=[image, TextContent(text=text)])


def test_to_openai_responses_dicts_handles_image_first_content():
    message = _user_message_with_image_first("hello world")
    serialized = Message.to_openai_responses_dicts_from_list([message])
    parts = serialized[0]["content"]
    assert any(part["type"] == "input_text" and part["text"] == "hello world" for part in parts)
    assert any(part["type"] == "input_image" for part in parts)


def test_fill_image_content_in_responses_input_includes_image_parts():
    message = _user_message_with_image_first("describe image")
    serialized = Message.to_openai_responses_dicts_from_list([message])
    rewritten = fill_image_content_in_responses_input(serialized, [message])
    assert rewritten == serialized


def test_to_openai_responses_dicts_handles_image_only_content():
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    message = Message(role=MessageRole.user, content=[image])
    serialized = Message.to_openai_responses_dicts_from_list([message])
    parts = serialized[0]["content"]
    assert parts[0]["type"] == "input_image"


def test_to_anthropic_dict_falls_back_for_malformed_tool_call_arguments():
    malformed_args = '{"message": "unterminated}'
    msg = Message(
        role=MessageRole.assistant,
        content=[TextContent(text="thinking")],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_test_malformed",
                type="function",
                function=Function(name="send_message", arguments=malformed_args),
            )
        ],
    )

    serialized = msg.to_anthropic_dict(
        current_model="anthropic/claude-sonnet-4-5-20250929",
        inner_thoughts_xml_tag="thinking",
        put_inner_thoughts_in_kwargs=False,
    )

    tool_use_items = [item for item in serialized["content"] if item.get("type") == "tool_use"]
    assert len(tool_use_items) == 1
    assert tool_use_items[0]["input"] == {"_malformed_tool_arguments": malformed_args}


def test_to_google_dict_falls_back_for_malformed_tool_call_arguments():
    malformed_args = '{"message": "unterminated}'
    msg = Message(
        role=MessageRole.assistant,
        content=[],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_test_malformed_google",
                type="function",
                function=Function(name="send_message", arguments=malformed_args),
            )
        ],
    )

    serialized = msg.to_google_dict(
        current_model="google/gemini-2.5-pro",
    )

    function_calls = [item for item in serialized["parts"] if item.get("functionCall")]
    assert len(function_calls) == 1
    assert function_calls[0]["functionCall"]["args"] == {"_malformed_tool_arguments": malformed_args}


def test_to_google_dict_preserves_thought_signature_on_empty_content():
    """When Gemini returns a function call without reasoning text, the
    thought_signature must still appear on the serialized functionCall part.
    Regression test for LET-8166 / GitHub #3221."""
    sig = "EoQHsomebase64signaturedata=="
    msg = Message(
        role=MessageRole.assistant,
        content=[TextContent(text="", signature=sig)],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_test_thought_sig",
                type="function",
                function=Function(name="archival_memory_search", arguments='{"query": "test"}'),
            )
        ],
    )

    serialized = msg.to_google_dict(current_model="google/gemini-3-flash")

    function_calls = [p for p in serialized["parts"] if "functionCall" in p]
    assert len(function_calls) == 1
    assert function_calls[0].get("thought_signature") == sig


def test_to_google_dict_no_signature_when_absent():
    """Without a signature, functionCall parts should not include
    thought_signature (no sentinel, no empty string)."""
    msg = Message(
        role=MessageRole.assistant,
        content=[],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_test_no_sig",
                type="function",
                function=Function(name="send_message", arguments='{"message": "hi"}'),
            )
        ],
    )

    serialized = msg.to_google_dict(current_model="google/gemini-3-flash")

    function_calls = [p for p in serialized["parts"] if "functionCall" in p]
    assert len(function_calls) == 1
    assert "thought_signature" not in function_calls[0]


# ---------------------------------------------------------------------------
# dedupe_tool_messages_for_llm_api
# ---------------------------------------------------------------------------
#
# Background: tool_call_ids from clients like Letta Code are not globally
# unique — counters reset per session and IDs collide across long histories.
# The dedup pass must therefore scope its uniqueness check to a single
# tool-batch (consecutive tool messages between two non-tool messages),
# never across batches. Otherwise the second batch's tool message gets
# wrongly dropped, leaving its assistant tool_call orphan and triggering a
# 400 from strict OpenAI-compat providers (Moonshot).


def _tool_msg_with_returns(call_ids, content_text="ok"):
    """Build a tool-role message with one explicit ToolReturn per call_id."""
    from letta.schemas.message import ToolReturn

    return Message(
        role=MessageRole.tool,
        content=[TextContent(text=content_text)],
        tool_returns=[
            ToolReturn(tool_call_id=cid, status="success", func_response="r")
            for cid in call_ids
        ],
    )


def _user_msg(text="hi"):
    return Message(role=MessageRole.user, content=[TextContent(text=text)])


def _assistant_with_calls(call_ids, content_text=""):
    return Message(
        role=MessageRole.assistant,
        content=[TextContent(text=content_text)],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=cid,
                type="function",
                function=Function(name="Read", arguments="{}"),
            )
            for cid in call_ids
        ],
    )


def test_dedupe_tool_messages_within_batch_drops_duplicates():
    """Within a single batch, duplicate tool_call_ids are deduped."""
    messages = [
        _user_msg(),
        _assistant_with_calls(["A", "B"]),
        _tool_msg_with_returns(["A", "A", "B"]),  # "A" duplicated
    ]
    result = Message.dedupe_tool_messages_for_llm_api(messages)
    tool_msgs = [m for m in result if m.role == MessageRole.tool]
    all_returns = [tr for m in tool_msgs for tr in (m.tool_returns or [])]
    ids = [tr.tool_call_id for tr in all_returns]
    assert ids.count("A") == 1
    assert ids.count("B") == 1


def test_dedupe_does_not_cross_batch_boundaries():
    """Same tool_call_id in two different batches must be preserved.

    This is the regression test for the chatlounge-agent / Read:206 bug.
    """
    messages = [
        _user_msg(),
        _assistant_with_calls(["Read:206"]),
        _tool_msg_with_returns(["Read:206"]),
        _user_msg("next turn"),  # batch boundary — resets dedup window
        _assistant_with_calls(["Read:206"]),  # collision: counter reused
        _tool_msg_with_returns(["Read:206"]),
    ]
    result = Message.dedupe_tool_messages_for_llm_api(messages)
    tool_msgs = [m for m in result if m.role == MessageRole.tool]
    # Both tool messages must survive — they're in different batches.
    assert len(tool_msgs) == 2, f"expected 2 tool msgs, got {len(tool_msgs)}"
    for m in tool_msgs:
        ids = [tr.tool_call_id for tr in (m.tool_returns or [])]
        assert "Read:206" in ids


def test_dedupe_preserves_order():
    messages = [
        _user_msg(),
        _assistant_with_calls(["A"]),
        _tool_msg_with_returns(["A"]),
        _user_msg("turn2"),
        _assistant_with_calls(["B"]),
        _tool_msg_with_returns(["B"]),
    ]
    result = Message.dedupe_tool_messages_for_llm_api(messages)
    roles = [m.role for m in result]
    assert roles == [
        MessageRole.user,
        MessageRole.assistant,
        MessageRole.tool,
        MessageRole.user,
        MessageRole.assistant,
        MessageRole.tool,
    ]


def test_dedupe_within_consecutive_tool_messages_in_same_batch():
    """Two consecutive tool messages (no non-tool between them) ARE same batch."""
    messages = [
        _user_msg(),
        _assistant_with_calls(["A", "B"]),
        _tool_msg_with_returns(["A"]),
        _tool_msg_with_returns(["A"]),  # duplicate of first batch's A — should drop
        _tool_msg_with_returns(["B"]),
    ]
    result = Message.dedupe_tool_messages_for_llm_api(messages)
    tool_msgs = [m for m in result if m.role == MessageRole.tool]
    flat_ids = [
        tr.tool_call_id for m in tool_msgs for tr in (m.tool_returns or [])
    ]
    assert flat_ids.count("A") == 1
    assert flat_ids.count("B") == 1


def test_dedupe_handles_empty_input():
    assert Message.dedupe_tool_messages_for_llm_api([]) == []


def test_dedupe_passes_through_when_no_tool_messages():
    messages = [_user_msg(), _assistant_with_calls(["A"])]
    result = Message.dedupe_tool_messages_for_llm_api(messages)
    assert len(result) == 2
