"""Test file-backed application configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

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
