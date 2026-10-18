"""Test Gemini report generation without making API requests."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

from google.genai import errors
from loguru import logger
import pytest

from ff_assistant.analysis import SYSTEM, analyze


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
    client_class.assert_called_once_with(api_key="test-key")
    request = create.call_args.kwargs
    assert request["model"] == "gemini-3.8-flash"
    assert request["system_instruction"] == SYSTEM
    assert '"id":"123"' in request["input"][0]["content"][0]["text"]
    assert request["tools"][0]["name"] == "get_player_stats"
    assert request["tools"][1]["name"] == "get_player_matchups"
    assert request["tools"][2]["name"] == "get_player_values"
    assert request["store"] is False
    assert request["generation_config"]["max_output_tokens"] == 1800


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
