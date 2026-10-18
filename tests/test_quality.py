"""Test report consistency review without contacting a model."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from langchain_google_genai.chat_models import GoogleRateLimitError

from ff_assistant.quality import ReportReviewer
from ff_assistant.request_throttle import ModelRequestThrottle


def test_reviewer_accepts_consistent_report() -> None:
    """Return no findings for a valid, consistent JSON verdict."""
    reviewer = ReportReviewer("test-key")
    assert reviewer._chain.steps[1].max_retries == 1
    reviewer._chain = Mock()
    reviewer._chain.invoke.return_value = {
        "consistent": True,
        "contradictions": [],
    }

    assert reviewer.review("Start the healthy player.") == ()
    assert reviewer._chain.invoke.call_args.args[0]["report"] == (
        "Start the healthy player."
    )


def test_reviewer_returns_specific_conflicts() -> None:
    """Return evidence for a report with conflicting claims."""
    reviewer = ReportReviewer("test-key")
    reviewer._chain = Mock()
    reviewer._chain.invoke.return_value = {
        "consistent": False,
        "contradictions": ["Calls Player A both active and out."],
    }

    assert reviewer.review("Player A is active. Player A is out.") == (
        "Calls Player A both active and out.",
    )


@pytest.mark.parametrize(
    "result",
    [
        {"consistent": True, "contradictions": ["Conflict"]},
        {"consistent": False, "contradictions": []},
        {"consistent": "yes", "contradictions": []},
        {"consistent": True, "contradictions": [123]},
        {},
    ],
)
def test_reviewer_rejects_invalid_verdict(result: dict[str, object]) -> None:
    """Fail closed when the model result is not a valid consistency verdict."""
    reviewer = ReportReviewer("test-key")
    reviewer._chain = Mock()
    reviewer._chain.invoke.return_value = result

    with pytest.raises(RuntimeError, match="invalid result"):
        reviewer.review("Draft report")


def test_reviewer_hides_request_details_on_failure() -> None:
    """Do not expose the API key or report through raised model errors."""
    reviewer = ReportReviewer("test-key", retry_delay_seconds=0)
    reviewer._chain = Mock()
    reviewer._chain.invoke.side_effect = TimeoutError("secret report test-key")

    with pytest.raises(
        RuntimeError, match="QA review failed \\(TimeoutError\\)"
    ) as raised:
        reviewer.review("secret report")

    assert raised.value.__cause__ is None
    assert "secret report" not in str(raised.value)
    assert "test-key" not in str(raised.value)


def test_reviewer_retries_rate_limit_with_shared_throttle() -> None:
    """Pause and reacquire the request slot before a second QA attempt."""
    throttle = Mock(spec=ModelRequestThrottle)
    reviewer = ReportReviewer(
        "test-key",
        retry_delay_seconds=1.25,
        request_throttle=throttle,
    )
    reviewer._chain = Mock()
    reviewer._chain.invoke.side_effect = [
        GoogleRateLimitError("Too many requests"),
        {"consistent": True, "contradictions": []},
    ]

    with patch("ff_assistant.quality.sleep") as sleep:
        assert reviewer.review("A consistent report") == ()

    assert throttle.wait.call_count == 2
    throttle.wait.assert_any_call("gemini-3.5-flash-lite")
    sleep.assert_called_once_with(1.25)
