"""Test player stat totals, efficiency, and snap joins without network access."""

from __future__ import annotations

from typing import Any

import pytest

from ff_assistant.player_stats import PlayerStatsLookup


class _Frame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def to_dicts(self) -> list[dict[str, Any]]:
        return self.rows


def test_lookup_calculates_totals_and_snap_percentage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bridge a Sleeper ESPN ID to weekly stats and PFR snap counts."""
    stats = [
        {
            "player_id": "g1",
            "player_display_name": "Example Runner",
            "season_type": "REG",
            "week": week,
            "game_id": f"game-{week}",
            "team": "EX",
            "attempts": 0,
            "completions": 0,
            "passing_yards": 0,
            "carries": carries,
            "rushing_yards": yards,
            "targets": targets,
            "receptions": catches,
            "receiving_yards": receiving_yards,
            "rushing_tds": 1 if week == 2 else 0,
            "receiving_tds": 0,
        }
        for week, carries, yards, targets, catches, receiving_yards in (
            (1, 10, 40, 4, 3, 30),
            (2, 20, 100, 6, 4, 40),
        )
    ]
    snaps = [
        {
            "game_type": "REG",
            "pfr_player_id": "p1",
            "game_id": f"game-{week}",
            "offense_pct": pct,
        }
        for week, pct in ((1, 0.5), (2, 0.75))
    ]
    monkeypatch.setattr(
        "ff_assistant.player_stats.nflreadpy.load_player_stats",
        lambda season, summary_level: _Frame(stats),
    )
    monkeypatch.setattr(
        "ff_assistant.player_stats.nflreadpy.load_players",
        lambda: _Frame([{"gsis_id": "g1", "pfr_id": "p1", "espn_id": "e1"}]),
    )
    monkeypatch.setattr(
        "ff_assistant.player_stats.nflreadpy.load_snap_counts",
        lambda season: _Frame(snaps),
    )
    lookup = PlayerStatsLookup(
        2026, {"s1": {"espn_id": "e1", "full_name": "Example Runner"}}
    )

    result = lookup.lookup(["s1"], recent_games=1)

    player = result["players"][0]
    assert result["latest_data_week"] == 2
    assert player["season_totals"]["rushing"]["carries"] == 30
    assert player["season_totals"]["rushing"]["yards_per_carry"] == 4.67
    assert player["season_totals"]["receiving"]["yards_per_target"] == 7.0
    assert player["season_totals"]["avg_offensive_snap_pct"] == 62.5
    assert player["recent_games"]["totals"]["rushing"]["yards_per_carry"] == 5.0
    assert player["recent_games"]["totals"]["avg_offensive_snap_pct"] == 75.0


def test_lookup_reports_unmatched_player_without_inventing_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an unavailable result for a player with no unambiguous match."""
    monkeypatch.setattr(
        "ff_assistant.player_stats.nflreadpy.load_player_stats",
        lambda season, summary_level: _Frame([]),
    )
    monkeypatch.setattr(
        "ff_assistant.player_stats.nflreadpy.load_players", lambda: _Frame([])
    )
    monkeypatch.setattr(
        "ff_assistant.player_stats.nflreadpy.load_snap_counts",
        lambda season: _Frame([]),
    )

    result = PlayerStatsLookup(2026, {"s1": {"full_name": "Unknown"}}).lookup(["s1"])

    assert result["players"][0]["status"] == "unavailable"
    assert "season_totals" not in result["players"][0]
