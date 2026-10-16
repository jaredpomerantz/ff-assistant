"""Look up nflverse player statistics for Sleeper players."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import nflreadpy  # type: ignore[import-untyped]
from loguru import logger

MAX_PLAYERS_PER_LOOKUP = 12
MAX_RECENT_GAMES = 5


def _total(rows: list[dict[str, Any]], field: str) -> int:
    return sum(int(row.get(field) or 0) for row in rows)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 2) if denominator else None


def _percentage(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 1) if denominator else None


def _summary(
    rows: list[dict[str, Any]],
    pfr_id: str | None,
    snaps_by_game: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    attempts = _total(rows, "attempts")
    completions = _total(rows, "completions")
    passing_yards = _total(rows, "passing_yards")
    carries = _total(rows, "carries")
    rushing_yards = _total(rows, "rushing_yards")
    targets = _total(rows, "targets")
    receptions = _total(rows, "receptions")
    receiving_yards = _total(rows, "receiving_yards")
    snap_rows = [
        snap
        for row in rows
        if pfr_id
        and (snap := snaps_by_game.get((pfr_id, str(row["game_id"])))) is not None
    ]
    snap_pcts = [
        float(snap["offense_pct"])
        for snap in snap_rows
        if snap.get("offense_pct") is not None
    ]
    return {
        "games_played": len(rows),
        "passing": {
            "attempts": attempts,
            "completions": completions,
            "yards": passing_yards,
            "touchdowns": _total(rows, "passing_tds"),
            "interceptions": _total(rows, "passing_interceptions"),
            "completion_pct": _percentage(completions, attempts),
            "yards_per_attempt": _ratio(passing_yards, attempts),
        },
        "rushing": {
            "carries": carries,
            "yards": rushing_yards,
            "touchdowns": _total(rows, "rushing_tds"),
            "yards_per_carry": _ratio(rushing_yards, carries),
            "carries_per_game": _ratio(carries, len(rows)),
        },
        "receiving": {
            "targets": targets,
            "receptions": receptions,
            "yards": receiving_yards,
            "touchdowns": _total(rows, "receiving_tds"),
            "catch_pct": _percentage(receptions, targets),
            "yards_per_target": _ratio(receiving_yards, targets),
            "yards_per_reception": _ratio(receiving_yards, receptions),
            "targets_per_game": _ratio(targets, len(rows)),
        },
        "yards_per_touch": _ratio(
            rushing_yards + receiving_yards, carries + receptions
        ),
        "snap_data_games": len(snap_pcts),
        "avg_offensive_snap_pct": (
            round(100 * sum(snap_pcts) / len(snap_pcts), 1) if snap_pcts else None
        ),
    }


class PlayerStatsLookup:
    """Load one season of stats lazily and serve bounded player lookups.

    Attributes:
        season: NFL season year used for every lookup.
    """

    def __init__(self, season: int, sleeper_players: dict[str, Any]) -> None:
        """Initialize the lookup without downloading data.

        Args:
            season: NFL season year.
            sleeper_players: Sleeper player map keyed by Sleeper player ID.
        """
        self.season = season
        self._sleeper_players = sleeper_players
        self._stats_by_gsis: dict[str, list[dict[str, Any]]] | None = None
        self._name_ids: dict[str, set[str]] = defaultdict(set)
        self._gsis_by_espn: dict[str, str] = {}
        self._pfr_by_gsis: dict[str, str] = {}
        self._snaps_by_game: dict[tuple[str, str], dict[str, Any]] = {}
        self._latest_data_week: int | None = None
        self._load_error: str | None = None

    def _load(self) -> None:
        try:
            frame = nflreadpy.load_player_stats(self.season, summary_level="week")
            stats_by_gsis: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in frame.to_dicts():
                if row.get("season_type") != "REG" or not row.get("player_id"):
                    continue
                gsis_id = str(row["player_id"])
                stats_by_gsis[gsis_id].append(row)
                if row.get("player_display_name"):
                    self._name_ids[str(row["player_display_name"]).casefold()].add(
                        gsis_id
                    )
                week = int(row["week"])
                self._latest_data_week = max(self._latest_data_week or week, week)
            self._stats_by_gsis = stats_by_gsis
            logger.info(
                "Loaded regular-season player stats through week {}",
                self._latest_data_week,
            )
        except Exception as exc:
            self._load_error = (
                f"Weekly player stats could not be loaded ({type(exc).__name__})"
            )
            logger.warning("{}", self._load_error)
            return

        try:
            espn_ids: dict[str, set[str]] = defaultdict(set)
            for row in nflreadpy.load_players().to_dicts():
                gsis_id = row.get("gsis_id")
                if not gsis_id:
                    continue
                gsis_id = str(gsis_id)
                if row.get("espn_id"):
                    espn_ids[str(row["espn_id"])].add(gsis_id)
                if row.get("pfr_id"):
                    self._pfr_by_gsis[gsis_id] = str(row["pfr_id"])
            self._gsis_by_espn = {
                espn_id: next(iter(ids))
                for espn_id, ids in espn_ids.items()
                if len(ids) == 1
            }
        except Exception as exc:
            logger.warning("Player ID mappings unavailable: {}", type(exc).__name__)

        try:
            for row in nflreadpy.load_snap_counts(self.season).to_dicts():
                if row.get("game_type") != "REG" or not row.get("pfr_player_id"):
                    continue
                key = (str(row["pfr_player_id"]), str(row["game_id"]))
                self._snaps_by_game[key] = row
        except Exception as exc:
            logger.warning("Offensive snap data unavailable: {}", type(exc).__name__)

    def _resolve_gsis_id(self, sleeper_id: str) -> str | None:
        player = self._sleeper_players.get(sleeper_id)
        if not isinstance(player, dict):
            return None
        assert self._stats_by_gsis is not None
        gsis_id = str(player.get("gsis_id") or "")
        if gsis_id in self._stats_by_gsis:
            return gsis_id
        espn_id = str(player.get("espn_id") or "")
        mapped_id = self._gsis_by_espn.get(espn_id)
        if mapped_id in self._stats_by_gsis:
            return mapped_id
        name = str(player.get("full_name") or "").casefold()
        candidates = self._name_ids.get(name, set())
        if len(candidates) == 1:
            return next(iter(candidates))
        team = player.get("team")
        matching_team = {
            candidate
            for candidate in candidates
            if any(row.get("team") == team for row in self._stats_by_gsis[candidate])
        }
        return next(iter(matching_team)) if len(matching_team) == 1 else None

    def lookup(self, player_ids: list[str], recent_games: int = 3) -> dict[str, Any]:
        """Return counting, volume, efficiency, and available snap stats.

        Args:
            player_ids: Sleeper IDs present in the report context.
            recent_games: Number of most recent played games to summarize.

        Returns:
            Stats for each player, with explicit unavailable results when needed.
        """
        if (
            not isinstance(player_ids, list)
            or not 1 <= len(player_ids) <= MAX_PLAYERS_PER_LOOKUP
            or any(not isinstance(player_id, str) for player_id in player_ids)
            or type(recent_games) is not int
            or not 1 <= recent_games <= MAX_RECENT_GAMES
        ):
            return {
                "status": "invalid_request",
                "reason": "Invalid player IDs or recent_games",
            }
        if self._stats_by_gsis is None and self._load_error is None:
            self._load()
        if self._load_error:
            return {"status": "unavailable", "reason": self._load_error}
        assert self._stats_by_gsis is not None

        results: list[dict[str, Any]] = []
        for sleeper_id in dict.fromkeys(player_ids):
            player = self._sleeper_players.get(sleeper_id)
            gsis_id = self._resolve_gsis_id(sleeper_id)
            if not gsis_id:
                results.append(
                    {
                        "sleeper_id": sleeper_id,
                        "status": "unavailable",
                        "reason": "No unambiguous current-season player stats match",
                    }
                )
                continue
            assert isinstance(player, dict)
            rows = sorted(
                self._stats_by_gsis[gsis_id], key=lambda row: int(row["week"])
            )
            recent = rows[-recent_games:]
            pfr_id = self._pfr_by_gsis.get(gsis_id)
            results.append(
                {
                    "sleeper_id": sleeper_id,
                    "status": "available",
                    "player_name": player.get("full_name"),
                    "team": rows[-1].get("team"),
                    "weeks_played": [int(row["week"]) for row in rows],
                    "season_totals": _summary(rows, pfr_id, self._snaps_by_game),
                    "recent_games": {
                        "weeks": [int(row["week"]) for row in recent],
                        "totals": _summary(recent, pfr_id, self._snaps_by_game),
                    },
                }
            )
        return {
            "status": "available",
            "source": "nflverse weekly player stats via nflreadpy; snaps from PFR",
            "season": self.season,
            "season_type": "regular",
            "latest_data_week": self._latest_data_week,
            "players": results,
        }
