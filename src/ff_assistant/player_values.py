"""Look up current dynasty player values from TheTradeCalc's public API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from time import monotonic
from typing import Any
from unicodedata import normalize
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

VALUES_URL = "https://thetradecalc.com/api/v1/values?mode=dynasty"
ATTRIBUTION = "[Values by TheTradeCalc](https://thetradecalc.com)"
MAX_PLAYERS_PER_LOOKUP = 12
CACHE_SECONDS = 600


def _name_key(name: str) -> str:
    """Normalize names without dropping suffixes that distinguish players."""
    plain = normalize("NFKD", name).casefold()
    return "".join(character for character in plain if character.isalnum())


class PlayerValuesLookup:
    """Fetch a bounded set of dynasty values, caching the public board briefly."""

    def __init__(self, sleeper_players: dict[str, Any]) -> None:
        """Keep Sleeper metadata for ID-to-name matching."""
        self._sleeper_players = sleeper_players
        self._board: dict[str, Any] | None = None
        self._loaded_at = 0.0

    def _load(self) -> dict[str, Any]:
        if self._board is not None and monotonic() - self._loaded_at < CACHE_SECONDS:
            return self._board
        request = Request(VALUES_URL, headers={"User-Agent": "ff-assistant/0.1"})
        with urlopen(request, timeout=15) as response:
            board = json.load(response)
        if (
            not isinstance(board, dict)
            or board.get("mode") != "dynasty"
            or not isinstance(board.get("values"), list)
            or not isinstance(board.get("generated"), str)
        ):
            raise ValueError("TheTradeCalc returned an invalid dynasty values feed")
        self._board = board
        self._loaded_at = monotonic()
        logger.info("Loaded {} TheTradeCalc dynasty values", len(board["values"]))
        return board

    def lookup(self, player_ids: list[str]) -> dict[str, Any]:
        """Return source values and ranks for Sleeper players, if unambiguous."""
        if not player_ids or len(player_ids) > MAX_PLAYERS_PER_LOOKUP:
            return {
                "status": "invalid_request",
                "reason": "Request one to twelve player IDs",
            }
        if any(not isinstance(player_id, str) for player_id in player_ids):
            return {"status": "invalid_request", "reason": "Player IDs must be strings"}
        try:
            board = self._load()
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            logger.warning("TheTradeCalc values unavailable: {}", type(exc).__name__)
            reason = f"TheTradeCalc values could not be loaded ({type(exc).__name__})"
            return {
                "status": "unavailable",
                "reason": reason,
            }

        by_name: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in board["values"]:
            if not isinstance(item, dict) or item.get("type") != "player":
                continue
            name = item.get("name")
            position = item.get("position")
            if isinstance(name, str) and isinstance(position, str):
                by_name.setdefault((_name_key(name), position), []).append(item)

        players: list[dict[str, Any]] = []
        for player_id in player_ids:
            sleeper = self._sleeper_players.get(player_id) or {}
            name = (
                sleeper.get("full_name")
                or (
                    f"{sleeper.get('first_name') or ''} "
                    f"{sleeper.get('last_name') or ''}"
                ).strip()
            )
            position = sleeper.get("position")
            if not isinstance(name, str) or not isinstance(position, str):
                players.append({"player_id": player_id, "status": "unavailable"})
                continue
            matches = by_name.get((_name_key(name), position), [])
            if len(matches) > 1:
                matches = [
                    item for item in matches if item.get("team") == sleeper.get("team")
                ]
            if len(matches) != 1:
                players.append({"player_id": player_id, "status": "unavailable"})
                continue
            item = matches[0]
            if not all(
                isinstance(item.get(field), int)
                for field in ("value", "rank", "positionRank")
            ):
                players.append({"player_id": player_id, "status": "unavailable"})
                continue
            players.append(
                {
                    "player_id": player_id,
                    "status": "available",
                    "name": item.get("name"),
                    "position": item.get("position"),
                    "team": item.get("team"),
                    "value": item.get("value"),
                    "overall_rank": item.get("rank"),
                    "position_rank": item.get("positionRank"),
                    "player_url": item.get("url"),
                }
            )
        return {
            "status": "available",
            "mode": "dynasty",
            "baseline": board.get("baseline"),
            "generated_at_utc": board["generated"],
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "attribution": ATTRIBUTION,
            "players": players,
        }
