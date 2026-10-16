"""Test matchup mapping and failure behavior without network requests."""

from __future__ import annotations

from typing import Any

import pytest

from ff_assistant.player_matchups import PlayerMatchupLookup


class _Frame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def to_dicts(self) -> list[dict[str, Any]]:
        return self.rows


def test_lookup_maps_opponents_byes_and_kickoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map Sleeper's Rams code and distinguish a real bye from an unknown team."""
    rows = [
        {
            "season": 2026,
            "game_type": "REG",
            "week": 6,
            "gameday": "2026-10-15",
            "gametime": "20:15",
            "home_team": "DEN",
            "away_team": "SEA",
            "location": "Home",
        },
        {
            "season": 2026,
            "game_type": "REG",
            "week": 6,
            "gameday": "2026-10-18",
            "gametime": "16:05",
            "home_team": "LA",
            "away_team": "ARI",
            "location": "Home",
        },
        {
            "season": 2026,
            "game_type": "REG",
            "week": 5,
            "home_team": "CIN",
            "away_team": "PIT",
        },
    ]
    monkeypatch.setattr(
        "ff_assistant.player_matchups.nflreadpy.load_schedules",
        lambda season: _Frame(rows),
    )
    players = {
        "seahawk": {"team": "SEA"},
        "ram": {"team": "LAR"},
        "bengal": {"team": "CIN"},
        "old": {"team": "OAK"},
    }
    lookup = PlayerMatchupLookup(2026, 6, "regular", players)

    result = lookup.lookup(list(players))

    assert result["status"] == "available"
    assert result["week"] == 6
    seahawk, ram, bengal, old = result["players"]
    assert seahawk["opponent"] == "DEN"
    assert seahawk["home_away"] == "away"
    assert seahawk["kickoff_utc"] == "2026-10-16T00:15+00:00"
    assert ram["team"] == "LAR"
    assert ram["opponent"] == "ARI"
    assert bengal["status"] == "bye"
    assert old["status"] == "unavailable"


def test_lookup_reports_missing_schedule_without_claiming_byes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid bye claims when the current-week schedule cannot be loaded."""

    def fail(_season: int) -> None:
        raise ConnectionError("schedule unavailable")

    monkeypatch.setattr("ff_assistant.player_matchups.nflreadpy.load_schedules", fail)

    result = PlayerMatchupLookup(2026, 6, "regular", {"s1": {"team": "SEA"}}).lookup(
        ["s1"]
    )

    assert result["status"] == "unavailable"
    assert "bye" not in str(result)
