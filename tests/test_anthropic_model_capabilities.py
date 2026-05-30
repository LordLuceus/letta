"""Tests for the model-capability predicates in `letta.llm_api.anthropic_client`.

These predicates decide which Anthropic API request shape to send for a given
model. Historically they were hardcoded with `startswith("claude-opus-4-7")`
checks, which broke immediately when Opus 4.8 shipped with identical API
constraints. These tests pin the version-comparison logic so new minor model
releases (4.8, 4.9, 5.x, ...) work without code changes.

Source of truth for the constraints:
- https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-8
- https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking.md

API constraints per model:
- Opus <= 4.5: legacy `thinking: {type: "enabled", budget_tokens: N}` only.
- Opus 4.6 / Sonnet 4.6: adaptive thinking via the `adaptive-thinking-2026-01-28`
  beta header; legacy thinking still works but deprecated.
- Opus 4.7+ (incl. 4.8, future minor versions): adaptive thinking is GA (no
  beta header). Legacy `type: "enabled"` is REJECTED with 400. Non-default
  temperature/top_p/top_k is REJECTED with 400. Effort is GA (no beta header).
"""

from letta.llm_api.anthropic_client import (
    _is_ga_adaptive,
    _needs_effort_beta,
    _rejects_sampling_params,
    _supports_effort,
    _uses_adaptive_thinking,
)


# ---------------------------------------------------------------------------
# _uses_adaptive_thinking — controls whether to send `type: "adaptive"` vs
# `type: "enabled" + budget_tokens` in the thinking block.
# ---------------------------------------------------------------------------


class TestUsesAdaptiveThinking:
    def test_opus_4_6_uses_adaptive(self):
        assert _uses_adaptive_thinking("claude-opus-4-6") is True
        assert _uses_adaptive_thinking("claude-opus-4-6-20251015") is True

    def test_sonnet_4_6_uses_adaptive(self):
        assert _uses_adaptive_thinking("claude-sonnet-4-6") is True
        assert _uses_adaptive_thinking("claude-sonnet-4-6-20251015") is True

    def test_opus_4_7_uses_adaptive(self):
        assert _uses_adaptive_thinking("claude-opus-4-7") is True

    def test_opus_4_8_uses_adaptive(self):
        # Regression: Opus 4.8 inherits 4.7's API shape. Must use adaptive
        # thinking; sending budget_tokens returns 400.
        assert _uses_adaptive_thinking("claude-opus-4-8") is True

    def test_future_opus_minor_versions_use_adaptive(self):
        # Forward-compat: Opus 4.9, 4.10, ... should not require a code patch.
        assert _uses_adaptive_thinking("claude-opus-4-9") is True
        assert _uses_adaptive_thinking("claude-opus-4-10") is True
        assert _uses_adaptive_thinking("claude-opus-4-15-20270301") is True

    def test_future_sonnet_minor_versions_use_adaptive(self):
        assert _uses_adaptive_thinking("claude-sonnet-4-7") is True
        assert _uses_adaptive_thinking("claude-sonnet-4-8") is True

    def test_future_opus_5_uses_adaptive(self):
        assert _uses_adaptive_thinking("claude-opus-5") is True
        assert _uses_adaptive_thinking("claude-opus-5-0") is True
        assert _uses_adaptive_thinking("claude-opus-5-1-20270601") is True

    def test_opus_4_5_does_not_use_adaptive(self):
        # Opus 4.5 only supports legacy budget_tokens thinking.
        assert _uses_adaptive_thinking("claude-opus-4-5") is False
        assert _uses_adaptive_thinking("claude-opus-4-5-20250930") is False

    def test_opus_4_1_does_not_use_adaptive(self):
        assert _uses_adaptive_thinking("claude-opus-4-1") is False

    def test_sonnet_4_5_does_not_use_adaptive(self):
        # Sonnet 4.5 predates adaptive thinking.
        assert _uses_adaptive_thinking("claude-sonnet-4-5") is False

    def test_sonnet_4_does_not_use_adaptive(self):
        assert _uses_adaptive_thinking("claude-sonnet-4") is False

    def test_haiku_does_not_use_adaptive(self):
        # Haiku currently has no adaptive-thinking support.
        assert _uses_adaptive_thinking("claude-haiku-4-5") is False

    def test_pre_4_models_do_not_use_adaptive(self):
        assert _uses_adaptive_thinking("claude-3-7-sonnet-20250219") is False
        assert _uses_adaptive_thinking("claude-3-5-sonnet-20240620") is False


# ---------------------------------------------------------------------------
# _is_ga_adaptive — controls whether to skip the `adaptive-thinking-2026-01-28`
# beta header. True iff adaptive thinking is GA for the model.
# ---------------------------------------------------------------------------


class TestIsGaAdaptive:
    def test_opus_4_7_is_ga(self):
        assert _is_ga_adaptive("claude-opus-4-7") is True

    def test_opus_4_8_is_ga(self):
        # Regression: 4.8 inherits GA status; sending the beta header is
        # harmless but redundant.
        assert _is_ga_adaptive("claude-opus-4-8") is True

    def test_future_opus_minor_versions_are_ga(self):
        assert _is_ga_adaptive("claude-opus-4-9") is True
        assert _is_ga_adaptive("claude-opus-5") is True

    def test_opus_4_6_is_not_ga(self):
        # 4.6 still needs the beta header.
        assert _is_ga_adaptive("claude-opus-4-6") is False

    def test_sonnet_4_6_is_not_ga(self):
        assert _is_ga_adaptive("claude-sonnet-4-6") is False

    def test_opus_4_5_is_not_ga(self):
        assert _is_ga_adaptive("claude-opus-4-5") is False


# ---------------------------------------------------------------------------
# _rejects_sampling_params — true iff sending temperature/top_p/top_k returns
# 400. We omit the param entirely in that case.
# ---------------------------------------------------------------------------


class TestRejectsSamplingParams:
    def test_opus_4_7_rejects(self):
        assert _rejects_sampling_params("claude-opus-4-7") is True

    def test_opus_4_8_rejects(self):
        # Regression: 4.8 inherits the sampling-param rejection.
        assert _rejects_sampling_params("claude-opus-4-8") is True

    def test_future_opus_minor_versions_reject(self):
        assert _rejects_sampling_params("claude-opus-4-9") is True
        assert _rejects_sampling_params("claude-opus-5") is True

    def test_opus_4_6_accepts(self):
        assert _rejects_sampling_params("claude-opus-4-6") is False

    def test_sonnet_4_6_accepts(self):
        assert _rejects_sampling_params("claude-sonnet-4-6") is False

    def test_opus_4_5_accepts(self):
        assert _rejects_sampling_params("claude-opus-4-5") is False

    def test_sonnet_4_5_accepts(self):
        assert _rejects_sampling_params("claude-sonnet-4-5") is False


# ---------------------------------------------------------------------------
# _supports_effort — true iff the model accepts `output_config.effort`.
# ---------------------------------------------------------------------------


class TestSupportsEffort:
    def test_opus_4_5_supports(self):
        assert _supports_effort("claude-opus-4-5") is True

    def test_opus_4_6_supports(self):
        assert _supports_effort("claude-opus-4-6") is True

    def test_opus_4_7_supports(self):
        assert _supports_effort("claude-opus-4-7") is True

    def test_opus_4_8_supports(self):
        # Regression: effort is GA on 4.7+; 4.8 inherits.
        assert _supports_effort("claude-opus-4-8") is True

    def test_sonnet_4_6_supports(self):
        assert _supports_effort("claude-sonnet-4-6") is True

    def test_future_opus_minor_versions_support(self):
        assert _supports_effort("claude-opus-4-9") is True
        assert _supports_effort("claude-opus-5") is True

    def test_opus_4_1_does_not_support(self):
        assert _supports_effort("claude-opus-4-1") is False

    def test_sonnet_4_5_does_not_support(self):
        assert _supports_effort("claude-sonnet-4-5") is False

    def test_haiku_does_not_support(self):
        assert _supports_effort("claude-haiku-4-5") is False


# ---------------------------------------------------------------------------
# _needs_effort_beta — true iff the model supports effort but it's still
# behind the `effort-2025-11-24` beta header (i.e. Opus 4.5/4.6, Sonnet 4.6).
# False on Opus 4.7+ (GA).
# ---------------------------------------------------------------------------


class TestNeedsEffortBeta:
    def test_opus_4_5_needs_beta(self):
        assert _needs_effort_beta("claude-opus-4-5") is True

    def test_opus_4_6_needs_beta(self):
        assert _needs_effort_beta("claude-opus-4-6") is True

    def test_sonnet_4_6_needs_beta(self):
        assert _needs_effort_beta("claude-sonnet-4-6") is True

    def test_opus_4_7_does_not_need_beta(self):
        assert _needs_effort_beta("claude-opus-4-7") is False

    def test_opus_4_8_does_not_need_beta(self):
        # Regression: GA on 4.7+ means 4.8 must not send the beta header.
        assert _needs_effort_beta("claude-opus-4-8") is False

    def test_future_opus_minor_versions_do_not_need_beta(self):
        assert _needs_effort_beta("claude-opus-4-9") is False
        assert _needs_effort_beta("claude-opus-5") is False

    def test_models_without_effort_do_not_need_beta(self):
        # If the model doesn't support effort at all, the beta question is
        # moot — return False.
        assert _needs_effort_beta("claude-3-5-sonnet-20240620") is False
        assert _needs_effort_beta("claude-haiku-4-5") is False


# ---------------------------------------------------------------------------
# Garbage / unknown model strings must not blow up. The helpers are called
# unconditionally for every request, so they have to be total functions.
# ---------------------------------------------------------------------------


class TestRobustnessAgainstUnknownInputs:
    UNKNOWN_INPUTS = [
        "",
        "gpt-4",
        "claude",
        "claude-opus",
        "claude-opus-",
        "claude-opus-4",
        "claude-opus-x",
        "claude-opus-4-x",
        "not-a-model-id",
        "anthropic/claude-opus-4-8",  # full handle (with provider prefix)
    ]

    def test_uses_adaptive_does_not_raise(self):
        for m in self.UNKNOWN_INPUTS:
            _uses_adaptive_thinking(m)  # must not raise

    def test_is_ga_adaptive_does_not_raise(self):
        for m in self.UNKNOWN_INPUTS:
            _is_ga_adaptive(m)

    def test_rejects_sampling_params_does_not_raise(self):
        for m in self.UNKNOWN_INPUTS:
            _rejects_sampling_params(m)

    def test_supports_effort_does_not_raise(self):
        for m in self.UNKNOWN_INPUTS:
            _supports_effort(m)

    def test_needs_effort_beta_does_not_raise(self):
        for m in self.UNKNOWN_INPUTS:
            _needs_effort_beta(m)
