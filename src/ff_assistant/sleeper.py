"""Small read-only client for Sleeper's documented public API."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from importlib.resources import files
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from loguru import logger

BASE_URL = "https://api.sleeper.app/v1"


class SleeperError(RuntimeError):
    """Indicate that Sleeper data could not be retrieved or used."""

    pass


def get_json(path: str, params: dict[str, str | int] | None = None) -> Any:
    """Fetch JSON from the Sleeper API.

    Args:
        path: API path relative to the Sleeper v1 endpoint.
        params: Optional query parameters.

    Returns:
        The decoded JSON response.

    Raises:
        SleeperError: The request fails or returns invalid JSON.
    """
    url = f"{BASE_URL}/{path.lstrip('/')}"
    if params:
        url += "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "ff-assistant/0.1"})
    try:
        with urlopen(request, timeout=25) as response:
            return json.load(response)
    except HTTPError as exc:
        raise SleeperError(
            f"Sleeper request for {path} failed (HTTP {exc.code})"
        ) from exc
    except (URLError, TimeoutError, ValueError) as exc:
        raise SleeperError(
            f"Sleeper request for {path} failed ({type(exc).__name__})"
        ) from exc


def get_user(username: str) -> dict[str, Any]:
    """Return the Sleeper user with the given username.

    Raises:
        SleeperError: The user does not exist or the API request fails.
    """
    user = get_json(f"user/{quote(username, safe='')}")
    if not user or not user.get("user_id"):
        raise SleeperError("Sleeper user was not found")
    return user


def get_state() -> dict[str, Any]:
    """Return the current NFL season and week from Sleeper."""
    return get_json("state/nfl")


def get_leagues(user_id: str, season: str) -> list[dict[str, Any]]:
    """Return a user's NFL leagues for one season."""
    return get_json(f"user/{user_id}/leagues/nfl/{season}") or []


def get_league(league_id: str) -> dict[str, Any]:
    """Return details for a Sleeper league."""
    return get_json(f"league/{league_id}")


def get_rosters(league_id: str) -> list[dict[str, Any]]:
    """Return every roster in a Sleeper league."""
    return get_json(f"league/{league_id}/rosters")


def get_trending(kind: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return trending NFL players for the requested transaction kind."""
    return get_json(
        f"players/nfl/trending/{kind}", {"lookback_hours": 72, "limit": limit}
    )


def get_players() -> dict[str, Any]:
    """Return the bundled Sleeper player snapshot.

    Returns:
        Player data keyed by Sleeper player ID.

    Raises:
        SleeperError: The bundled snapshot is missing or invalid.
    """
    resource = files("ff_assistant").joinpath("resources/sleeper-players.json.gz")
    try:
        with resource.open("rb") as compressed:
            with gzip.open(compressed, "rt", encoding="utf-8") as file:
                players = json.load(file)
    except (OSError, EOFError, ValueError) as exc:
        raise SleeperError(
            "Bundled Sleeper player snapshot is unavailable or invalid"
        ) from exc
    if not isinstance(players, dict):
        raise SleeperError("Bundled Sleeper player snapshot must be a player map")
    logger.info("Loaded {} players from bundled Sleeper snapshot", len(players))
    return players


def get_players_fetched_at() -> str:
    """Return the UTC fetch time recorded for the bundled player snapshot."""
    resource = files("ff_assistant").joinpath(
        "resources/sleeper-players-fetched-at.txt"
    )
    try:
        return resource.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SleeperError("Bundled Sleeper snapshot timestamp is unavailable") from exc


def player_summary(player_id: str, players: dict[str, Any]) -> dict[str, Any]:
    """Return the roster and injury fields for one player."""
    raw = players.get(player_id) or {}
    return {
        "id": player_id,
        "name": raw.get("full_name")
        or (
            f"{raw.get('first_name') or ''} {raw.get('last_name') or ''}".strip()
            or player_id
        ),
        "position": raw.get("position") or ("DEF" if len(player_id) <= 3 else None),
        "team": raw.get("team"),
        "status": raw.get("status"),
        "injury_status": raw.get("injury_status"),
        "injury_body_part": raw.get("injury_body_part"),
        "depth_chart_order": raw.get("depth_chart_order"),
        "bye_week": raw.get("bye_week"),
    }


def build_context(
    league: dict[str, Any],
    user_id: str,
    players: dict[str, Any],
    trending: list[dict[str, Any]],
    fetched_at: str,
) -> dict[str, Any]:
    """Build a report snapshot for the user's roster and available players.

    Args:
        league: Sleeper league details.
        user_id: Sleeper user ID of the roster owner or co-owner.
        players: Player data keyed by Sleeper player ID.
        trending: Recently added players from Sleeper.
        fetched_at: UTC timestamp for the snapshot.

    Returns:
        League, roster, waiver, source, and limitation data for a report.

    Raises:
        SleeperError: The user has no roster in the league.
    """
    league_id = str(league["league_id"])
    rosters = get_rosters(league_id)
    mine = next(
        (
            r
            for r in rosters
            if str(r.get("owner_id")) == user_id
            or user_id in (r.get("co_owners") or [])
        ),
        None,
    )
    if mine is None:
        raise SleeperError(f"No roster for user in league {league_id}")
    owned = {pid for roster in rosters for pid in (roster.get("players") or [])}
    my_ids = mine.get("players") or []
    trending_candidates = []
    for item in trending:
        pid = str(item.get("player_id"))
        if pid in owned:
            continue
        summary = player_summary(pid, players)
        if summary["position"] not in {"QB", "RB", "WR", "TE", "K", "DEF"}:
            continue
        if summary["status"] in {"Inactive", "Retired"}:
            continue
        summary["trending_adds_72h"] = item.get("count")
        trending_candidates.append(summary)
        if len(trending_candidates) == 35:
            break
    starters = mine.get("starters") or []
    reserve = set(mine.get("reserve") or [])
    taxi = set(mine.get("taxi") or [])
    settings = league.get("settings") or {}
    faab_enabled = settings.get("waiver_type") == 2
    faab_used = mine.get("settings", {}).get("waiver_budget_used") or 0
    return {
        "fetched_at_utc": fetched_at,
        "data_sources": {
            "league_and_players": (
                "Sleeper public API; player metadata from bundled snapshot fetched "
                f"{get_players_fetched_at()}"
            ),
            "injury_status": (
                "Bundled Sleeper player metadata; may be stale and is not an "
                "official NFL injury report"
            ),
            "projections": "unavailable",
            "matchups": "nflverse schedule through get_player_matchups tool",
            "official_injury_report_reference": "https://www.nfl.com/injuries/",
        },
        "league": {
            "id": league_id,
            "name": league.get("name"),
            "season": league.get("season"),
            "status": league.get("status"),
            "roster_positions": league.get("roster_positions"),
            "scoring_settings": league.get("scoring_settings"),
            "settings": {
                key: settings.get(key)
                for key in (
                    "waiver_type",
                    "waiver_budget",
                    "daily_waivers",
                    "best_ball",
                    "reserve_slots",
                )
            },
            "faab_enabled": faab_enabled,
        },
        "my_roster": {
            "roster_id": mine.get("roster_id"),
            "players": [
                {
                    **player_summary(pid, players),
                    "current_slot": (
                        "starter"
                        if pid in starters
                        else "reserve"
                        if pid in reserve
                        else "taxi"
                        if pid in taxi
                        else "bench"
                    ),
                }
                for pid in my_ids
            ],
            "current_starters_in_slot_order": [
                player_summary(pid, players) for pid in starters
            ],
            "faab_remaining": (
                settings.get("waiver_budget", 0) - faab_used if faab_enabled else None
            ),
        },
        "available_trending_adds": trending_candidates,
        "limitations": [
            "Bundled Sleeper player metadata is only as current as its recorded "
            "fetch time; roster names, teams, status, and injury fields may be stale.",
            "No verified weekly projections or injury change history are "
            "integrated. Use get_player_matchups before stating opponents, byes, "
            "or kickoff times, and do not infer projected points from opponents. "
            "Only state practice participation when an available official "
            "injury report supplies it.",
            "Trending additions show popularity, not fantasy value. This pool "
            "may miss useful players who are not trending.",
        ],
    }


def utc_now() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
