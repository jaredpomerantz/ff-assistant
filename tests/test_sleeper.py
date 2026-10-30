"""Test trending coverage and refreshes of the local player directory."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from ff_assistant import sleeper
from ff_assistant.refresh_players import refresh_players


def test_trending_requests_broader_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request 200 players by default while allowing a custom limit."""
    get_json = Mock(return_value=[])
    monkeypatch.setattr(sleeper, "get_json", get_json)

    sleeper.get_trending("add")
    get_json.assert_called_once_with(
        "players/nfl/trending/add", {"lookback_hours": 72, "limit": 200}
    )
    sleeper.get_trending("add", limit=75)
    assert get_json.call_args.args[1]["limit"] == 75


def test_context_keeps_deeper_eligible_trending_players(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep free agents beyond the old limits after filtering owned players."""
    player_ids = [str(index) for index in range(1000, 1200)]
    monkeypatch.setattr(
        sleeper,
        "get_rosters",
        lambda _: [
            {"owner_id": "me", "players": player_ids[:35]},
            {"owner_id": "other", "players": player_ids[35:70]},
        ],
    )
    players = {
        player_id: {"full_name": player_id, "position": "RB", "status": "Active"}
        for player_id in player_ids
    }
    players[player_ids[70]]["status"] = "Retired"
    players[player_ids[71]]["position"] = "LB"
    del players[player_ids[72]]

    context = sleeper.build_context(
        {"league_id": "1"},
        "me",
        players,
        [
            {"player_id": player_id, "count": 200 - i}
            for i, player_id in enumerate(player_ids)
        ],
        "now",
    )

    candidates = context["available_trending_adds"]
    assert [player["id"] for player in candidates] == player_ids[73:]
    assert candidates[-1]["trending_adds_72h"] == 1


def test_refresh_writes_snapshot_consumed_by_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load newly fetched players and their timestamp through existing readers."""
    players = {"12345": {"full_name": "New Player", "position": "WR"}}
    get_json = Mock(return_value=players)
    monkeypatch.setattr(sleeper, "get_json", get_json)
    monkeypatch.setattr(sleeper, "utc_now", lambda: "2026-10-30T13:00:00Z")
    monkeypatch.setattr(sleeper, "files", lambda _: tmp_path)

    refresh_players(tmp_path / "resources")

    get_json.assert_called_once_with("players/nfl")
    assert sleeper.get_players() == players
    assert sleeper.get_players_fetched_at() == "2026-10-30T13:00:00Z"
    assert len(list((tmp_path / "resources").iterdir())) == 2


@pytest.mark.parametrize("response", [None, [], {}, {"12345": "invalid"}])
def test_invalid_refresh_preserves_existing_resources(
    response: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not replace a working snapshot or timestamp with an invalid response."""
    snapshot = tmp_path / "sleeper-players.json.gz"
    timestamp = tmp_path / "sleeper-players-fetched-at.txt"
    snapshot.write_bytes(b"existing snapshot")
    timestamp.write_text("existing timestamp", encoding="utf-8")
    monkeypatch.setattr(sleeper, "get_json", Mock(return_value=response))

    with pytest.raises(sleeper.SleeperError, match="nonempty player map"):
        refresh_players(tmp_path)

    assert snapshot.read_bytes() == b"existing snapshot"
    assert timestamp.read_text(encoding="utf-8") == "existing timestamp"


def test_failed_refresh_preserves_existing_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate fetch errors so a job cannot silently report with stale data."""
    snapshot = tmp_path / "sleeper-players.json.gz"
    snapshot.write_bytes(b"existing snapshot")
    monkeypatch.setattr(
        sleeper, "get_json", Mock(side_effect=sleeper.SleeperError("fetch failed"))
    )

    with pytest.raises(sleeper.SleeperError, match="fetch failed"):
        refresh_players(tmp_path)

    assert snapshot.read_bytes() == b"existing snapshot"
