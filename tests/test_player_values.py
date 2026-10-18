"""Test TheTradeCalc player value lookups without network access."""

from __future__ import annotations

import io
import json
from unittest.mock import Mock

import pytest

from ff_assistant.player_values import PlayerValuesLookup


def _board() -> dict[str, object]:
    return {
        "mode": "dynasty",
        "generated": "2026-10-18T18:06:02Z",
        "baseline": "Superflex, 12 teams, PPR, no TE premium",
        "values": [
            {
                "type": "player",
                "name": "Jahmyr Gibbs",
                "position": "RB",
                "team": "DET",
                "value": 9999,
                "rank": 1,
                "positionRank": 1,
                "url": "https://thetradecalc.com/dynasty/player/rb_jahmyrgibbs",
            },
            {"type": "pick", "name": "2027 Early 1st", "position": "PICK"},
        ],
    }


def test_lookup_matches_player_and_reuses_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map Sleeper IDs to rankings and cache the downloaded board."""
    response = io.BytesIO(json.dumps(_board()).encode())
    urlopen = Mock(return_value=response)
    monkeypatch.setattr("ff_assistant.player_values.urlopen", urlopen)
    lookup = PlayerValuesLookup(
        {"s1": {"full_name": "Jahmyr Gibbs", "position": "RB", "team": "DET"}}
    )

    first = lookup.lookup(["s1", "missing"])
    second = lookup.lookup(["s1"])

    assert first["players"][0]["value"] == 9999
    assert first["players"][0]["overall_rank"] == 1
    assert first["players"][0]["position_rank"] == 1
    assert first["players"][1]["status"] == "unavailable"
    assert first["generated_at_utc"] == "2026-10-18T18:06:02Z"
    assert first["attribution"].startswith("[Values by TheTradeCalc]")
    assert second["players"][0]["value"] == 9999
    urlopen.assert_called_once()


def test_lookup_reports_api_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return unavailable data without failing the rest of a report."""
    monkeypatch.setattr(
        "ff_assistant.player_values.urlopen",
        Mock(side_effect=TimeoutError("network failed")),
    )
    result = PlayerValuesLookup({}).lookup(["s1"])

    assert result["status"] == "unavailable"
    assert "TimeoutError" in result["reason"]


def test_lookup_rejects_ambiguous_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid assigning one player's value to a namesake on another team."""
    board = _board()
    values = board["values"]
    assert isinstance(values, list)
    values.append(
        {
            "type": "player",
            "name": "Jahmyr Gibbs",
            "position": "RB",
            "team": "OTHER",
            "value": 500,
            "rank": 100,
            "positionRank": 30,
        }
    )
    monkeypatch.setattr(
        "ff_assistant.player_values.urlopen",
        Mock(return_value=io.BytesIO(json.dumps(board).encode())),
    )
    lookup = PlayerValuesLookup({"s1": {"full_name": "Jahmyr Gibbs", "position": "RB"}})

    assert lookup.lookup(["s1"])["players"][0]["status"] == "unavailable"
