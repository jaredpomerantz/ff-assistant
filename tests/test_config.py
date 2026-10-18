"""Test file-backed application configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

from loguru import logger
import pytest

from ff_assistant import main
from ff_assistant.config import load_config


def test_load_config_ignores_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use dotenv values even when process variables have the same names."""
    path = tmp_path / ".env"
    path.write_text('GEMINI_API_KEY="file-key"\n', encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "process-key")

    assert load_config(path)["GEMINI_API_KEY"] == "file-key"


def test_load_config_requires_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a missing dotenv file even if the process has a key."""
    monkeypatch.setenv("GEMINI_API_KEY", "process-key")

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(tmp_path / ".env")


def test_run_uses_dotenv_username(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read the Sleeper username from the file, not the process environment."""
    (tmp_path / ".env").write_text(
        'SLEEPER_USERNAME="file-user"\nSLEEPER_SEASON="2026"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLEEPER_USERNAME", "process-user")
    monkeypatch.setattr(sys, "argv", ["ff-assistant", "--list-leagues"])
    get_user = Mock(return_value={"user_id": "123"})
    monkeypatch.setattr(main.sleeper, "get_user", get_user)
    monkeypatch.setattr(main.sleeper, "get_leagues", lambda *_: [])

    assert main.run() == 0
    get_user.assert_called_once_with("file-user")


def test_main_enables_debug_level_from_dotenv() -> None:
    """Configure the stderr logger before running a DEBUG report."""
    config = {"LOG_LEVEL": "DEBUG"}
    with (
        patch.object(main, "load_config", return_value=config),
        patch.object(main, "run", return_value=0) as run,
        patch.object(main, "logger") as log,
    ):
        with pytest.raises(SystemExit) as raised:
            main.main()

    assert raised.value.code == 0
    assert log.add.call_args.kwargs["level"] == "DEBUG"
    run.assert_called_once_with(config)


def test_run_identifies_failed_report_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Name the league and failed step in logs and the Discord notification."""
    (tmp_path / ".env").write_text(
        'SLEEPER_USERNAME="file-user"\n'
        'SLEEPER_SEASON="2026"\n'
        'GEMINI_API_KEY="secret-key"\n'
        'GEMINI_RETRY_DELAY_SECONDS="1.25"\n'
        'DISCORD_WEBHOOK_URL="https://example.com/webhook"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ff-assistant", "--mode", "waivers"])
    monkeypatch.setattr(main.sleeper, "get_user", lambda *_: {"user_id": "123"})
    monkeypatch.setattr(
        main.sleeper,
        "get_leagues",
        lambda *_: [{"league_id": "456", "name": "My League", "status": "in_season"}],
    )
    monkeypatch.setattr(main.sleeper, "get_players", lambda: {})
    monkeypatch.setattr(main.sleeper, "get_trending", lambda *_: [])
    monkeypatch.setattr(
        main.sleeper,
        "get_state",
        lambda: {"season_type": "off", "week": 1},
    )
    monkeypatch.setattr(
        main.sleeper,
        "build_context",
        lambda *_: {"my_roster": {"players": []}},
    )
    monkeypatch.setattr(main, "PlayerStatsLookup", Mock())
    monkeypatch.setattr(main, "analyze", Mock(side_effect=RuntimeError("model busy")))
    send_report = Mock()
    monkeypatch.setattr(main, "send_report", send_report)
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        with pytest.raises(RuntimeError, match="Reports failed.*456"):
            main.run()
    finally:
        logger.remove(sink_id)

    assert any(
        "Failed waivers report for league My League (456) during generating" in message
        for message in messages
    )
    assert "while generating the report with Gemini" in send_report.call_args.args[1]
    assert main.analyze.call_args.kwargs["retry_delay_seconds"] == 1.25
    assert all("secret-key" not in message for message in messages)
