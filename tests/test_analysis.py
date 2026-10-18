"""Test Gemini report generation without making API requests."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

from google.genai import errors
from loguru import logger
import pytest

from ff_assistant.analysis import SYSTEM, analyze
from ff_assistant.quality import ReportReviewer
from ff_assistant.request_throttle import ModelRequestThrottle


def test_analyze_sends_report_to_gemini() -> None:
    """Pass the key, prompt, model, and stateless setting to Gemini."""
    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.return_value.output_text = "  Start this player.  "
        create.return_value.steps = []

        report = analyze(
            "lineup",
            {"league": {"id": "123"}},
            "test-key",
            "gemini-3.8-flash",
            Mock(),
            Mock(),
        )

    assert report == "Start this player."
    client_class.assert_called_once()
    assert client_class.call_args.kwargs["api_key"] == "test-key"
    assert client_class.call_args.kwargs["http_options"].retry_options.attempts == 1
    request = create.call_args.kwargs
    assert request["model"] == "gemini-3.8-flash"
    assert request["system_instruction"] == SYSTEM
    assert '"id":"123"' in request["input"][0]["content"][0]["text"]
    assert request["tools"][0]["name"] == "get_player_stats"
    assert request["tools"][1]["name"] == "get_player_matchups"
    assert request["tools"][2]["name"] == "get_player_values"
    assert request["store"] is False
    assert request["generation_config"]["max_output_tokens"] == 8192


def test_analyze_logs_model_output_only_at_debug() -> None:
    """Keep model text out of INFO logs while exposing it at DEBUG."""
    info_messages: list[str] = []
    debug_messages: list[str] = []
    info_sink = logger.add(
        lambda message: info_messages.append(message.record["message"]), level="INFO"
    )
    debug_sink = logger.add(
        lambda message: debug_messages.append(message.record["message"]),
        level="DEBUG",
    )
    try:
        with patch("ff_assistant.analysis.genai.Client") as client_class:
            create = (
                client_class.return_value.__enter__.return_value.interactions.create
            )
            create.return_value.output_text = "Private roster advice"
            create.return_value.steps = []
            create.return_value.status = "completed"
            create.return_value.id = "interaction-123"
            create.return_value.usage = Mock(
                total_input_tokens=100,
                total_output_tokens=20,
                total_thought_tokens=5,
            )

            assert (
                analyze("lineup", {}, "test-key", "gemini-3.8-flash", Mock(), Mock())
                == "Private roster advice"
            )
    finally:
        logger.remove(info_sink)
        logger.remove(debug_sink)

    assert all("Private roster advice" not in message for message in info_messages)
    assert all("Gemini turn 1 returned" not in message for message in info_messages)
    assert any(
        "Gemini turn 1 model output:\nPrivate roster advice" in message
        for message in debug_messages
    )
    assert any(
        "input_tokens=100 output_tokens=20" in message for message in debug_messages
    )


def test_analyze_rejects_empty_gemini_response() -> None:
    """Reject a Gemini response without report text."""
    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.return_value.output_text = None
        create.return_value.steps = []
        create.return_value.status = "failed"
        create.return_value.id = "interaction-123"
        create.return_value.usage = Mock(total_output_tokens=1800)
        create.return_value.errors = [Mock(code="MODEL_BUSY")]

        with pytest.raises(
            RuntimeError,
            match="status=failed.*interaction_id=interaction-123.*MODEL_BUSY.*output_tokens=1800",
        ):
            analyze("waivers", {}, "test-key", "gemini-3.8-flash", Mock(), Mock())

    assert create.call_args.kwargs["generation_config"]["max_output_tokens"] == 1800


def test_lineup_can_complete_after_five_tool_rounds() -> None:
    """Allow a complex lineup report to finish beyond the former four-round cap."""
    call = Mock()
    call.type = "function_call"
    call.name = "get_player_matchups"
    call.id = "call-1"
    call.arguments = {"player_ids": ["s1"]}
    call.model_dump.return_value = {
        "type": "function_call",
        "name": call.name,
        "id": call.id,
        "arguments": call.arguments,
    }
    lookup = Mock()
    lookup.lookup.return_value = {"status": "available", "players": []}
    context = {"my_roster": {"players": [{"id": "s1"}]}}
    responses = [Mock(steps=[call], output_text=None) for _ in range(5)]
    responses.append(Mock(steps=[], output_text="Keep the current starters."))

    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.side_effect = responses
        report = analyze(
            "lineup", context, "test-key", "gemini-3.8-flash", Mock(), lookup
        )

    assert report == "Keep the current starters."
    assert create.call_count == 6


def test_analyze_revises_and_rechecks_contradictory_report() -> None:
    """Do not return the first draft when QA identifies a contradiction."""
    reviewer = Mock()
    reviewer.review.side_effect = [("A is both active and out",), ()]
    first = Mock(steps=[], output_text="A is active. A is out.")
    revised = Mock(steps=[], output_text="A is out. Start B instead.")

    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.side_effect = [first, revised]
        report = analyze(
            "lineup",
            {},
            "test-key",
            "gemini-3.8-flash",
            Mock(),
            Mock(),
            qa_reviewer=reviewer,
        )

    assert report == "A is out. Start B instead."
    assert reviewer.review.call_count == 2
    assert create.call_count == 2
    revision_request = create.call_args.kwargs
    assert revision_request["tools"] == []
    revision_text = revision_request["input"][-1]["content"][0]["text"]
    assert "A is both active and out" in revision_text
    assert "A is active. A is out." in revision_text


def test_analyze_returns_report_only_after_qa_approval() -> None:
    """Review a clean draft once before returning it for delivery."""
    reviewer = Mock()
    reviewer.review.return_value = ()
    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.return_value.steps = []
        create.return_value.output_text = "Keep the current starters."

        report = analyze(
            "lineup",
            {},
            "test-key",
            "gemini-3.8-flash",
            Mock(),
            Mock(),
            qa_reviewer=reviewer,
        )

    assert report == "Keep the current starters."
    reviewer.review.assert_called_once_with(report)
    create.assert_called_once()


def test_main_turn_and_qa_review_share_request_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait five seconds between main and QA calls to the same model."""
    now = [100.0]
    waits: list[float] = []

    def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    monkeypatch.setattr("ff_assistant.request_throttle.monotonic", lambda: now[0])
    monkeypatch.setattr("ff_assistant.request_throttle.sleep", fake_sleep)
    throttle = ModelRequestThrottle(5.0)
    reviewer = ReportReviewer("test-key", request_throttle=throttle)
    reviewer._chain = Mock()
    reviewer._chain.invoke.return_value = {
        "consistent": True,
        "contradictions": [],
    }

    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.return_value.steps = []
        create.return_value.output_text = "Keep the current starters."
        report = analyze(
            "lineup",
            {},
            "test-key",
            "gemini-3.5-flash-lite",
            Mock(),
            Mock(),
            qa_reviewer=reviewer,
            request_throttle=throttle,
        )

    assert report == "Keep the current starters."
    assert waits == [5.0]


def test_qa_revision_and_recheck_keep_same_request_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Space draft, first QA, revision, and second QA through one timer."""
    now = [100.0]
    waits: list[float] = []

    def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    monkeypatch.setattr("ff_assistant.request_throttle.monotonic", lambda: now[0])
    monkeypatch.setattr("ff_assistant.request_throttle.sleep", fake_sleep)
    throttle = ModelRequestThrottle(5.0)
    reviewer = ReportReviewer("test-key", request_throttle=throttle)
    reviewer._chain = Mock()
    reviewer._chain.invoke.side_effect = [
        {"consistent": False, "contradictions": ["A is both active and out"]},
        {"consistent": True, "contradictions": []},
    ]
    draft = Mock(steps=[], output_text="A is active. A is out.")
    revision = Mock(steps=[], output_text="A is out.")

    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.side_effect = [draft, revision]
        report = analyze(
            "lineup",
            {},
            "test-key",
            "gemini-3.5-flash-lite",
            Mock(),
            Mock(),
            qa_reviewer=reviewer,
            request_throttle=throttle,
        )

    assert report == "A is out."
    assert waits == [5.0, 5.0, 5.0]


def test_analyze_withholds_report_when_revision_still_contradicts() -> None:
    """Fail the report after one unsuccessful correction."""
    reviewer = Mock()
    reviewer.review.side_effect = [("conflict",), ("conflict remains",)]
    first = Mock(steps=[], output_text="First draft")
    revised = Mock(steps=[], output_text="Still contradictory")

    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.side_effect = [first, revised]
        with pytest.raises(RuntimeError, match="unresolved contradictions"):
            analyze(
                "waivers",
                {},
                "test-key",
                "gemini-3.8-flash",
                Mock(),
                Mock(),
                qa_reviewer=reviewer,
            )


def test_analyze_logs_gemini_api_failure_without_request_content() -> None:
    """Report HTTP status and request ID without logging SDK response details."""
    response = Mock(headers={"x-request-id": "request-123"})
    failure = errors.ServerError(
        503,
        {"error": {"status": "UNAVAILABLE", "message": "roster-secret test-key"}},
        response,
    )
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        with (
            patch("ff_assistant.analysis.genai.Client") as client_class,
            patch("ff_assistant.analysis.sleep") as sleep,
        ):
            create = (
                client_class.return_value.__enter__.return_value.interactions.create
            )
            create.side_effect = failure
            with pytest.raises(
                RuntimeError,
                match="Gemini turn 1 failed \\(HTTP 503, status=UNAVAILABLE, "
                "request_id=request-123\\)",
            ) as raised:
                analyze(
                    "lineup",
                    {"private": "roster-secret"},
                    "test-key",
                    "gemini-3.8-flash",
                    Mock(),
                    Mock(),
                )
            assert create.call_count == 3
            assert sleep.call_count == 2
            sleep.assert_any_call(5.0)
    finally:
        logger.remove(sink_id)

    assert raised.value.__cause__ is None
    assert any("http_status=503" in message for message in messages)
    assert any("request_id=request-123" in message for message in messages)
    assert all("roster-secret" not in message for message in messages)
    assert all("test-key" not in message for message in messages)


def test_analyze_logs_transport_failure_type_without_exception_text() -> None:
    """Identify transport failures without exposing exception text."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        with (
            patch("ff_assistant.analysis.genai.Client") as client_class,
            patch("ff_assistant.analysis.sleep") as sleep,
        ):
            create = (
                client_class.return_value.__enter__.return_value.interactions.create
            )
            create.side_effect = TimeoutError("test-key")
            with pytest.raises(RuntimeError, match="turn 1 failed \\(TimeoutError\\)"):
                analyze("lineup", {}, "test-key", "gemini-3.8-flash", Mock(), Mock())
            assert create.call_count == 3
            assert sleep.call_count == 2
    finally:
        logger.remove(sink_id)

    assert any("error_type=TimeoutError" in message for message in messages)
    assert all("test-key" not in message for message in messages)


@pytest.mark.parametrize(
    ("code", "status"),
    [(429, "RESOURCE_EXHAUSTED"), (503, "UNAVAILABLE")],
)
def test_analyze_retries_transient_error_after_configured_delay(
    code: int, status: str
) -> None:
    """Retry a transient Gemini error using the configured delay."""
    failure = errors.APIError(code, {"error": {"status": status}})
    response = Mock(steps=[], output_text="Start this player.")
    with (
        patch("ff_assistant.analysis.genai.Client") as client_class,
        patch("ff_assistant.analysis.sleep") as sleep,
    ):
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.side_effect = [failure, response]

        report = analyze(
            "lineup", {}, "test-key", "gemini-3.8-flash", Mock(), Mock(), 1.25
        )

    assert report == "Start this player."
    assert create.call_count == 2
    sleep.assert_called_once_with(1.25)


def test_analyze_does_not_retry_invalid_request() -> None:
    """Return a permanent API error without sleeping or repeating the request."""
    failure = errors.ClientError(400, {"error": {"status": "INVALID_ARGUMENT"}})
    with (
        patch("ff_assistant.analysis.genai.Client") as client_class,
        patch("ff_assistant.analysis.sleep") as sleep,
    ):
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.side_effect = failure

        with pytest.raises(RuntimeError, match="HTTP 400"):
            analyze("lineup", {}, "test-key", "gemini-3.8-flash", Mock(), Mock())

    create.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize("delay", [-1.0, float("inf"), float("nan")])
def test_analyze_rejects_invalid_retry_delay(delay: float) -> None:
    """Reject negative and nonfinite retry delays before creating a client."""
    with patch("ff_assistant.analysis.genai.Client") as client_class:
        with pytest.raises(ValueError, match="nonnegative finite"):
            analyze("lineup", {}, "test-key", "gemini-3.8-flash", Mock(), Mock(), delay)

    client_class.assert_not_called()


def test_analyze_returns_player_stats_to_model() -> None:
    """Continue a stateless interaction with the requested player stats."""
    call = Mock()
    call.type = "function_call"
    call.name = "get_player_stats"
    call.id = "call-1"
    call.arguments = {"player_ids": ["s1"], "recent_games": 3}
    call.model_dump.return_value = {
        "type": "function_call",
        "name": call.name,
        "id": call.id,
        "arguments": call.arguments,
    }
    first = Mock(steps=[call], output_text=None)
    final = Mock(steps=[], output_text="Player had 10 targets.")
    stats_lookup = Mock()
    stats_lookup.lookup.return_value = {"status": "available", "targets": 10}
    context = {"my_roster": {"players": [{"id": "s1"}]}}

    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.side_effect = [first, final]
        report = analyze(
            "lineup", context, "test-key", "gemini-3.8-flash", stats_lookup, Mock()
        )

    assert report == "Player had 10 targets."
    stats_lookup.lookup.assert_called_once_with(["s1"], 3)
    history = create.call_args.kwargs["input"]
    assert history[1] == call.model_dump.return_value
    assert history[2]["type"] == "function_result"
    assert history[2]["call_id"] == "call-1"
    assert json.loads(history[2]["result"][0]["text"])["targets"] == 10


def test_analyze_returns_player_values_and_adds_attribution() -> None:
    """Expose only report players and credit TheTradeCalc in the final report."""
    calls = []
    for call_id, player_id in (("call-1", "s1"), ("call-2", "other")):
        call = Mock()
        call.type = "function_call"
        call.name = "get_player_values"
        call.id = call_id
        call.arguments = {"player_ids": [player_id]}
        call.model_dump.return_value = {
            "type": "function_call",
            "name": call.name,
            "id": call.id,
            "arguments": call.arguments,
        }
        calls.append(call)
    first = Mock(steps=calls, output_text=None)
    separator = "-" * 85
    final = Mock(steps=[], output_text=f"Gibbs ranks first.\n{separator}")
    values_lookup = Mock()
    values_lookup.lookup.return_value = {
        "status": "available",
        "players": [{"status": "available", "value": 9999}],
    }
    context = {"my_roster": {"players": [{"id": "s1"}]}}

    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.side_effect = [first, final]
        report = analyze(
            "lineup",
            context,
            "test-key",
            "gemini-3.8-flash",
            Mock(),
            Mock(),
            values_lookup=values_lookup,
        )

    values_lookup.lookup.assert_called_once_with(["s1"])
    history = create.call_args.kwargs["input"]
    assert json.loads(history[3]["result"][0]["text"])["players"][0]["value"] == 9999
    assert json.loads(history[4]["result"][0]["text"])["status"] == "invalid_request"
    assert "[Values by TheTradeCalc](https://thetradecalc.com)" in report
    assert report.endswith(separator)


def test_analyze_returns_matchups_only_for_report_players() -> None:
    """Expose schedule results for roster players without allowing arbitrary IDs."""
    calls = []
    for call_id, player_id in (("call-1", "s1"), ("call-2", "other")):
        call = Mock()
        call.type = "function_call"
        call.name = "get_player_matchups"
        call.id = call_id
        call.arguments = {"player_ids": [player_id]}
        call.model_dump.return_value = {
            "type": "function_call",
            "name": call.name,
            "id": call.id,
            "arguments": call.arguments,
        }
        calls.append(call)
    first = Mock(steps=calls, output_text=None)
    final = Mock(steps=[], output_text="The starter plays Denver.")
    matchups_lookup = Mock()
    matchups_lookup.lookup.return_value = {
        "status": "available",
        "players": [{"opponent": "DEN"}],
    }
    context = {"my_roster": {"players": [{"id": "s1"}]}}

    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.side_effect = [first, final]
        report = analyze(
            "lineup", context, "test-key", "gemini-3.8-flash", Mock(), matchups_lookup
        )

    assert report == "The starter plays Denver."
    matchups_lookup.lookup.assert_called_once_with(["s1"])
    history = create.call_args.kwargs["input"]
    assert (
        json.loads(history[3]["result"][0]["text"])["players"][0]["opponent"] == "DEN"
    )
    assert json.loads(history[4]["result"][0]["text"])["status"] == "invalid_request"
