"""Test Gemini report generation without making API requests."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from ff_assistant.analysis import SYSTEM, analyze


def test_analyze_sends_report_to_gemini() -> None:
    """Pass the key, prompt, model, and stateless setting to Gemini."""
    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.return_value.output_text = "  Start this player.  "
        create.return_value.steps = []

        report = analyze(
            "lineup", {"league": {"id": "123"}}, "test-key", "gemini-3.8-flash", Mock()
        )

    assert report == "Start this player."
    client_class.assert_called_once_with(api_key="test-key")
    request = create.call_args.kwargs
    assert request["model"] == "gemini-3.8-flash"
    assert request["system_instruction"] == SYSTEM
    assert '"id":"123"' in request["input"][0]["content"][0]["text"]
    assert request["tools"][0]["name"] == "get_player_stats"
    assert request["store"] is False
    assert request["generation_config"]["max_output_tokens"] == 1800


def test_analyze_rejects_empty_gemini_response() -> None:
    """Reject a Gemini response without report text."""
    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.return_value.output_text = None
        create.return_value.steps = []

        with pytest.raises(RuntimeError, match="empty report"):
            analyze("waivers", {}, "test-key", "gemini-3.8-flash", Mock())


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
            "lineup", context, "test-key", "gemini-3.8-flash", stats_lookup
        )

    assert report == "Player had 10 targets."
    stats_lookup.lookup.assert_called_once_with(["s1"], 3)
    history = create.call_args.kwargs["input"]
    assert history[1] == call.model_dump.return_value
    assert history[2]["type"] == "function_result"
    assert history[2]["call_id"] == "call-1"
    assert json.loads(history[2]["result"][0]["text"])["targets"] == 10
