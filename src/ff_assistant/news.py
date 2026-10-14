"""Read the official NFL injury table, rejecting stale week pages."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.request import Request, urlopen

from loguru import logger


class _InjuryTable(HTMLParser):
    """Collect player rows from the detailed NFL injury table."""

    def __init__(self) -> None:
        """Initialize parser state for the injury table."""
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_parts: list[str] = []
        self.cells: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track table, row, and cell opening tags."""
        if tag == "table" and "d3-o-reports--detailed" in (
            dict(attrs).get("class") or ""
        ):
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.cells = []
        elif self.in_row and tag == "td":
            self.in_cell = True
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        """Collect text inside a table cell."""
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Store completed cells and player rows."""
        if tag == "td" and self.in_cell:
            self.cells.append(" ".join(" ".join(self.cell_parts).split()))
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if len(self.cells) == 5:
                self.rows.append(self.cells)
            self.in_row = False
        elif tag == "table" and self.in_table:
            self.in_table = False


def parse_injury_page(html: str, requested_week: int) -> list[dict[str, str]]:
    """Parse the NFL injury table for the requested week.

    Args:
        html: Injury page markup.
        requested_week: Week expected in the page heading.

    Returns:
        Player injury rows from the detailed table.

    Raises:
        ValueError: The page is for another week or has no injury rows.
    """
    match = re.search(r"Injuries\s*-\s*WEEK\s*(\d+)", html, re.IGNORECASE)
    if not match or int(match.group(1)) != requested_week:
        raise ValueError("NFL injury page is for a different week")
    parser = _InjuryTable()
    parser.feed(html)
    if not parser.rows:
        raise ValueError("NFL injury table was empty")
    return [
        {
            "name": name,
            "position": position,
            "injury": injury,
            "practice_status": practice,
            "game_status": game,
        }
        for name, position, injury, practice, game in parser.rows
    ]


def fetch_injury_report(season: str, week: int) -> dict[str, Any]:
    """Fetch the official injury report, marking unavailable data explicitly.

    Args:
        season: NFL season year.
        week: Regular-season week.

    Returns:
        Report metadata and player rows, or an unavailable status.
    """
    url = f"https://www.nfl.com/injuries/league/{season}/reg{week}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 ff-assistant/0.1"})
    try:
        with urlopen(request, timeout=25) as response:
            html = response.read().decode("utf-8")
        rows = parse_injury_page(html, week)
        logger.info("Fetched {} official injury rows for week {}", len(rows), week)
        return {
            "status": "available",
            "url": url,
            "week": week,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "players": rows,
        }
    except Exception as exc:
        logger.warning(
            "Official injury report unavailable for week {}: {}",
            week,
            type(exc).__name__,
        )
        return {
            "status": "unavailable",
            "url": url,
            "week": week,
            "reason": (
                "Official report is unavailable or does not match the requested week"
            ),
        }


def for_roster(
    report: dict[str, Any], roster_players: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep official injury rows for players on the given roster.

    Args:
        report: Fetched official injury report.
        roster_players: Players on the user's roster.

    Returns:
        Report metadata with matching player rows when available.
    """
    if report["status"] != "available":
        return report
    names = {str(player["name"]).casefold() for player in roster_players}
    return {
        **{key: value for key, value in report.items() if key != "players"},
        "players": [
            row for row in report["players"] if row["name"].casefold() in names
        ],
    }
