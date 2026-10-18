"""Look up current-week NFL opponents and kickoff times for Sleeper players."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import nflreadpy  # type: ignore[import-untyped]
from loguru import logger

MAX_PLAYERS_PER_LOOKUP = 12
SLEEPER_TO_SCHEDULE_TEAM = {"LAR": "LA"}
SCHEDULE_TO_SLEEPER_TEAM = {"LA": "LAR"}
EASTERN = ZoneInfo("America/New_York")


def _kickoff(game: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return an Eastern kickoff label and its UTC instant when both are valid."""
    gameday = game.get("gameday")
    gametime = game.get("gametime")
    if not isinstance(gameday, str) or not isinstance(gametime, str):
        return None, None
    try:
        eastern = datetime.fromisoformat(f"{gameday}T{gametime}").replace(
            tzinfo=EASTERN
        )
    except ValueError:
        return None, None
    return eastern.isoformat(timespec="minutes"), eastern.astimezone(
        timezone.utc
    ).isoformat(timespec="minutes")


class PlayerMatchupLookup:
    """Load one week's schedule lazily and match it to supplied Sleeper player IDs."""

    def __init__(
        self,
        season: int,
        week: int,
        season_type: str,
        sleeper_players: dict[str, Any],
    ) -> None:
        """Store the week and player snapshot without requesting schedule data."""
        self.season = season
        self.week = week
        self.season_type = season_type
        self._sleeper_players = sleeper_players
        self._games_by_team: dict[str, dict[str, Any]] | None = None
        self._season_teams: set[str] = set()
        self._load_error: str | None = None
        self._retrieved_at_utc: str | None = None

    def _load(self) -> None:
        if self.season_type != "regular":
            self._load_error = "Only regular-season matchups are supported"
            return
        try:
            rows = nflreadpy.load_schedules(self.season).to_dicts()
            regular = [
                row
                for row in rows
                if row.get("season") == self.season and row.get("game_type") == "REG"
            ]
            self._season_teams = {
                str(team)
                for row in regular
                for team in (row.get("home_team"), row.get("away_team"))
                if team
            }
            games = [row for row in regular if row.get("week") == self.week]
            if not games:
                raise ValueError("No games found for the requested week")
            by_team: dict[str, dict[str, Any]] = {}
            for game in games:
                home = game.get("home_team")
                away = game.get("away_team")
                if (
                    not isinstance(home, str)
                    or not isinstance(away, str)
                    or not home
                    or not away
                    or home in by_team
                    or away in by_team
                ):
                    raise ValueError("The weekly schedule is incomplete or duplicated")
                by_team[home] = game
                by_team[away] = game
            self._games_by_team = by_team
            self._retrieved_at_utc = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            logger.info(
                "Loaded {} NFL games for {} week {}", len(games), self.season, self.week
            )
        except Exception as exc:
            self._load_error = (
                f"NFL schedule could not be loaded ({type(exc).__name__})"
            )
            logger.warning("{}", self._load_error)

    def lookup(self, player_ids: list[str]) -> dict[str, Any]:
        """Return scheduled opponents, kickoffs, byes, or unavailable status."""
        if (
            not isinstance(player_ids, list)
            or not 1 <= len(player_ids) <= MAX_PLAYERS_PER_LOOKUP
            or any(not isinstance(player_id, str) for player_id in player_ids)
        ):
            return {"status": "invalid_request", "reason": "Invalid player IDs"}
        if self._games_by_team is None and self._load_error is None:
            self._load()
        if self._load_error:
            return {"status": "unavailable", "reason": self._load_error}
        assert self._games_by_team is not None

        results: list[dict[str, Any]] = []
        for sleeper_id in dict.fromkeys(player_ids):
            player = self._sleeper_players.get(sleeper_id)
            team = player.get("team") if isinstance(player, dict) else None
            if not isinstance(team, str) or not team:
                results.append(
                    {
                        "sleeper_id": sleeper_id,
                        "status": "unavailable",
                        "reason": "No team in player snapshot",
                    }
                )
                continue
            schedule_team = SLEEPER_TO_SCHEDULE_TEAM.get(team, team)
            game = self._games_by_team.get(schedule_team)
            if game is None:
                results.append(
                    {
                        "sleeper_id": sleeper_id,
                        "team": team,
                        "status": "bye"
                        if schedule_team in self._season_teams
                        else "unavailable",
                    }
                )
                continue
            home = schedule_team == game["home_team"]
            opponent = game["away_team"] if home else game["home_team"]
            kickoff_eastern, kickoff_utc = _kickoff(game)
            results.append(
                {
                    "sleeper_id": sleeper_id,
                    "team": team,
                    "status": "scheduled",
                    "opponent": SCHEDULE_TO_SLEEPER_TEAM.get(opponent, opponent),
                    "home_away": "home" if home else "away",
                    "neutral_site": game.get("location") == "Neutral",
                    "kickoff_eastern": kickoff_eastern,
                    "kickoff_utc": kickoff_utc,
                }
            )
        return {
            "status": "available",
            "source": (
                "nflverse schedule via nflreadpy; player teams from Sleeper snapshot"
            ),
            "season": self.season,
            "week": self.week,
            "retrieved_at_utc": self._retrieved_at_utc,
            "players": results,
        }
