"""Test Gemini report generation without making API requests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ff_assistant.analysis import SYSTEM, analyze


def test_analyze_sends_report_to_gemini() -> None:
    """Pass the key, prompt, model, and stateless setting to Gemini."""
    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.return_value.output_text = "  Start this player.  "

        report = analyze(
            "lineup", {"league": {"id": "123"}}, "test-key", "gemini-3.8-flash"
        )

    assert report == "Start this player."
    client_class.assert_called_once_with(api_key="test-key")
    request = create.call_args.kwargs
    assert request["model"] == "gemini-3.8-flash"
    assert request["system_instruction"] == SYSTEM
    assert '"id":"123"' in request["input"]
    assert request["store"] is False
    assert request["generation_config"]["max_output_tokens"] == 1800


def test_analyze_rejects_empty_gemini_response() -> None:
    """Reject a Gemini response without report text."""
    with patch("ff_assistant.analysis.genai.Client") as client_class:
        create = client_class.return_value.__enter__.return_value.interactions.create
        create.return_value.output_text = None

        with pytest.raises(RuntimeError, match="empty report"):
            analyze("waivers", {}, "test-key", "gemini-3.8-flash")
