"""Test league selection, report preparation, and message delivery helpers."""

import pytest

from ff_assistant import sleeper
from ff_assistant.discord import split_report
from ff_assistant.main import selected_leagues
from ff_assistant.news import parse_injury_page
from ff_assistant.sleeper import build_context


def test_selects_all_active_leagues_or_requested_subset() -> None:
    """Select active leagues by default and requested IDs when supplied."""
    leagues = [
        {"league_id": "1", "status": "in_season"},
        {"league_id": "2", "status": "in_season"},
        {"league_id": "3", "status": "complete"},
    ]
    assert [league["league_id"] for league in selected_leagues(leagues, None)] == [
        "1",
        "2",
    ]
    assert [league["league_id"] for league in selected_leagues(leagues, "2,3")] == [
        "2",
        "3",
    ]


def test_get_players_uses_bundled_snapshot_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the bundled player map without requesting Sleeper's large endpoint."""
    monkeypatch.setattr(
        sleeper,
        "get_json",
        lambda *_: pytest.fail("Bundled player lookup made a network request"),
    )

    players = sleeper.get_players()

    assert len(players) > 1000
    assert sleeper.get_players_fetched_at().endswith("Z")


def test_waiver_pool_excludes_every_rostered_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclude players rostered by any team from waiver candidates."""
    monkeypatch.setattr(
        "ff_assistant.sleeper.get_rosters",
        lambda _: [
            {"owner_id": "me", "roster_id": 1, "players": ["a"], "starters": ["a"]},
            {"owner_id": "other", "roster_id": 2, "players": ["b"]},
        ],
    )
    players = {
        pid: {"full_name": pid, "position": "RB", "status": "Active"}
        for pid in ("a", "b", "c")
    }
    league = {
        "league_id": "1",
        "roster_positions": ["RB"],
        "scoring_settings": {"rec": 1},
    }
    context = build_context(
        league,
        "me",
        players,
        [{"player_id": pid, "count": 10} for pid in ("a", "b", "c")],
        "now",
    )
    assert [player["id"] for player in context["available_trending_adds"]] == ["c"]


def test_discord_chunks_fit_limit() -> None:
    """Keep every chunk below Discord's limit without dropping text."""
    chunks = split_report("A" * 4500)
    assert "".join(chunks) == "A" * 4500
    assert all(len(chunk) <= 1900 for chunk in chunks)


def test_official_report_rejects_stale_week() -> None:
    """Accept the requested week and reject a mismatched injury report."""
    html = (
        '<h2>Injuries - WEEK 5</h2><table class="d3-o-reports--detailed">'
        "<tr><td>Player</td><td>RB</td><td>Ankle</td><td>Limited</td>"
        "<td>Questionable</td></tr></table>"
    )
    assert parse_injury_page(html, 5)[0]["game_status"] == "Questionable"

    with pytest.raises(ValueError, match="different week"):
        parse_injury_page(html, 6)
