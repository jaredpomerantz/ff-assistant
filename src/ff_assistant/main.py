"""Run one independent report per Sleeper league."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ff_assistant import sleeper
from ff_assistant.analysis import analyze
from ff_assistant.config import load_config
from ff_assistant.discord import send_report
from ff_assistant.news import fetch_injury_report, for_roster


def selected_leagues(
    leagues: list[dict[str, Any]], league_ids: str | None
) -> list[dict[str, Any]]:
    """Select active leagues or the leagues explicitly requested.

    Args:
        leagues: Leagues available to the Sleeper user.
        league_ids: Comma-separated league IDs, or None for active leagues.

    Returns:
        The selected leagues in their original order.

    Raises:
        ValueError: A requested league ID is unavailable.
    """
    if not league_ids:
        return [league for league in leagues if league.get("status") == "in_season"]
    wanted = {item.strip() for item in league_ids.split(",") if item.strip()}
    found = [league for league in leagues if str(league.get("league_id")) in wanted]
    missing = wanted - {str(league["league_id"]) for league in found}
    if missing:
        raise ValueError("One or more requested league IDs were not found")
    return found


def run() -> int:
    """Fetch league data and generate, preview, or send the requested reports.

    Returns:
        Zero when all requested reports succeed.

    Raises:
        FileNotFoundError: The repository's .env file is missing.
        RuntimeError: Required settings or reports are unavailable.
    """
    parser = argparse.ArgumentParser(
        description="Fantasy football reports for Sleeper leagues"
    )
    parser.add_argument(
        "--mode",
        choices=("waivers", "injuries", "lineup"),
    )
    parser.add_argument("--list-leagues", action="store_true")
    parser.add_argument(
        "--preview-context",
        action="store_true",
        help="Fetch data and print a concise summary without calling OpenAI or Discord",
    )
    args = parser.parse_args()
    config = load_config(Path(".env"))
    mode = args.mode or config.get("TASK_MODE") or "lineup"
    if mode not in ("waivers", "injuries", "lineup"):
        parser.error(f"invalid TASK_MODE in .env: {mode}")

    username = config.get("SLEEPER_USERNAME")
    if not username:
        raise RuntimeError("SLEEPER_USERNAME is required in .env")
    user = sleeper.get_user(username)
    season = config.get("SLEEPER_SEASON") or str(sleeper.get_state()["season"])
    leagues = sleeper.get_leagues(str(user["user_id"]), season)
    if args.list_leagues:
        for league in leagues:
            print(f"{league['league_id']}  {league['name']}  ({league['status']})")
        return 0
    chosen = selected_leagues(leagues, config.get("SLEEPER_LEAGUE_IDS"))
    if not chosen:
        raise RuntimeError("No active leagues found for the configured user and season")

    webhook = config.get("DISCORD_WEBHOOK_URL", "")
    if not args.preview_context and not webhook:
        raise RuntimeError("DISCORD_WEBHOOK_URL is required in .env")
    api_key = config.get("OPENAI_API_KEY", "")
    if not args.preview_context and not api_key:
        raise RuntimeError("OPENAI_API_KEY is required in .env")

    cache = Path(config.get("SLEEPER_CACHE_PATH") or ".cache/sleeper-players.json")
    players = sleeper.get_players(cache)
    trending = sleeper.get_trending("add") if mode == "waivers" else []
    state = sleeper.get_state()
    official = (
        fetch_injury_report(season, int(state["week"]))
        if mode in ("injuries", "lineup") and state.get("season_type") == "regular"
        else {
            "status": "unavailable",
            "reason": "No current regular-season injury report",
        }
    )
    failures = []
    for number, league in enumerate(chosen, start=1):
        name = str(league.get("name") or league["league_id"])
        try:
            context = sleeper.build_context(
                league, str(user["user_id"]), players, trending, sleeper.utc_now()
            )
            context["official_injury_report"] = for_roster(
                official, context["my_roster"]["players"]
            )
            if args.preview_context:
                print(
                    json.dumps(
                        {
                            "league": name,
                            "roster_count": len(context["my_roster"]["players"]),
                            "trending_available_count": len(
                                context["available_trending_adds"]
                            ),
                        },
                        indent=2,
                    )
                )
                continue
            report = analyze(
                mode,
                context,
                api_key=api_key,
                model=config.get("OPENAI_MODEL") or "gpt-5.4-mini",
            )
            send_report(webhook, f"**{name} — {mode.title()} report**\n{report}")
            print(f"Sent {mode} report {number}/{len(chosen)}")
        except Exception as exc:
            failures.append(number)
            print(
                f"Failed to generate or send report {number}/{len(chosen)}: {type(exc).__name__}",
                file=sys.stderr,
            )
            if webhook and not args.preview_context:
                try:
                    send_report(
                        webhook,
                        f"Fantasy report failed for {name} ({mode}). Check the GitHub Actions run.",
                    )
                except Exception:
                    pass
    if failures:
        raise RuntimeError(
            f"Reports failed for {len(failures)} of {len(chosen)} leagues"
        )
    return 0


def main() -> None:
    """Run the command-line program and report failures on stderr."""
    try:
        raise SystemExit(run())
    except Exception as exc:
        print(f"Fantasy report failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
