"""Refresh the local Sleeper player directory before generating reports."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger

from ff_assistant import sleeper


def refresh_players(resources_dir: Path) -> None:
    """Fetch and replace the player snapshot and its UTC fetch timestamp.

    Args:
        resources_dir: Directory containing the application's player resources.

    Raises:
        sleeper.SleeperError: Fetching fails or the response is not a player map.
        OSError: Writing the refreshed resources fails.
    """
    logger.info("Refreshing Sleeper player snapshot")
    players = sleeper.get_json("players/nfl")
    if (
        not isinstance(players, dict)
        or not players
        or not all(isinstance(player, dict) for player in players.values())
    ):
        raise sleeper.SleeperError(
            "Sleeper player response must be a nonempty player map"
        )
    fetched_at = sleeper.utc_now()
    compressed = gzip.compress(json.dumps(players).encode("utf-8"), mtime=0)
    resources_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=resources_dir) as temporary_dir:
        temporary = Path(temporary_dir)
        snapshot = temporary / "sleeper-players.json.gz"
        timestamp = temporary / "sleeper-players-fetched-at.txt"
        snapshot.write_bytes(compressed)
        timestamp.write_text(f"{fetched_at}\n", encoding="utf-8")
        snapshot.replace(resources_dir / snapshot.name)
        timestamp.replace(resources_dir / timestamp.name)
    logger.info("Saved {} Sleeper players fetched at {}", len(players), fetched_at)


def main() -> None:
    """Refresh player resources in a source checkout from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resources-dir",
        type=Path,
        default=Path("src/ff_assistant/resources"),
        help="Output resource directory (default: src/ff_assistant/resources)",
    )
    args = parser.parse_args()
    refresh_players(args.resources_dir)


if __name__ == "__main__":
    main()
