"""Run one independent report per Sleeper league."""

from __future__ import annotations

import argparse
import json
import sys
from math import isfinite
from pathlib import Path
from typing import Any

from loguru import logger

from ff_assistant import sleeper
from ff_assistant.analysis import analyze
from ff_assistant.config import load_config
from ff_assistant.discord import send_report
from ff_assistant.news import fetch_injury_report, for_roster
from ff_assistant.player_matchups import PlayerMatchupLookup
from ff_assistant.player_stats import PlayerStatsLookup
from ff_assistant.player_values import PlayerValuesLookup


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


def run(config: dict[str, str] | None = None) -> int:
    """Fetch league data and generate, preview, or send the requested reports.

    Args:
        config: Settings loaded from .env, or None to load them here.

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
        help="Fetch data and print a concise summary without calling Gemini or Discord",
    )
    args = parser.parse_args()
    if config is None:
        config = load_config(Path(".env"))
    mode = args.mode or config.get("TASK_MODE") or "lineup"
    if mode not in ("waivers", "injuries", "lineup"):
        parser.error(f"invalid TASK_MODE in .env: {mode}")

    username = config.get("SLEEPER_USERNAME")
    if not username:
        raise RuntimeError("SLEEPER_USERNAME is required in .env")
    user = sleeper.get_user(username)
    season = config.get("SLEEPER_SEASON") or str(sleeper.get_state()["season"])
    logger.info("Fetching leagues for season {}", season)
    leagues = sleeper.get_leagues(str(user["user_id"]), season)
    if args.list_leagues:
        for league in leagues:
            print(f"{league['league_id']}  {league['name']}  ({league['status']})")
        return 0
    chosen = selected_leagues(leagues, config.get("SLEEPER_LEAGUE_IDS"))
    if not chosen:
        raise RuntimeError("No active leagues found for the configured user and season")
    logger.info("Selected {} leagues for {} reports", len(chosen), mode)

    webhook = config.get("DISCORD_WEBHOOK_URL", "")
    if not args.preview_context and not webhook:
        raise RuntimeError("DISCORD_WEBHOOK_URL is required in .env")
    api_key = config.get("GEMINI_API_KEY", "")
    if not args.preview_context and not api_key:
        raise RuntimeError("GEMINI_API_KEY is required in .env")
    try:
        retry_delay_seconds = float(config.get("GEMINI_RETRY_DELAY_SECONDS") or "5")
    except ValueError as exc:
        raise ValueError(
            "GEMINI_RETRY_DELAY_SECONDS must be a nonnegative finite number"
        ) from exc
    if not isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
        raise ValueError(
            "GEMINI_RETRY_DELAY_SECONDS must be a nonnegative finite number"
        )

    players = sleeper.get_players()
    stats_lookup = PlayerStatsLookup(int(season), players)
    values_lookup = PlayerValuesLookup(players)
    trending = sleeper.get_trending("add") if mode == "waivers" else []
    state = sleeper.get_state()
    matchups_lookup = PlayerMatchupLookup(
        int(season), int(state["week"]), str(state.get("season_type") or ""), players
    )
    official = (
        fetch_injury_report(season, int(state["week"]))
        if mode in ("injuries", "lineup") and state.get("season_type") == "regular"
        else {
            "status": "unavailable",
            "reason": "No current regular-season injury report",
        }
    )
    failures = []
    model = config.get("GEMINI_MODEL") or "gemini-3.5-flash-lite"
    for number, league in enumerate(chosen, start=1):
        league_id = str(league["league_id"])
        name = str(league.get("name") or league_id)
        stage = "building roster context"
        logger.info(
            "Preparing {} report for league {}/{}: {} ({})",
            mode,
            number,
            len(chosen),
            name,
            league_id,
        )
        try:
            context = sleeper.build_context(
                league, str(user["user_id"]), players, trending, sleeper.utc_now()
            )
            context["nfl_week"] = int(state["week"])
            context["season_type"] = state.get("season_type")
            stage = "matching the official injury report"
            context["official_injury_report"] = for_roster(
                official, context["my_roster"]["players"]
            )
            if args.preview_context:
                stage = "printing context preview"
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
                logger.info("Previewed league {}/{}", number, len(chosen))
                continue
            stage = f"generating the report with Gemini model {model}"
            logger.info("{} for league {} ({})", stage, name, league_id)
            report = analyze(
                mode,
                context,
                api_key=api_key,
                model=model,
                stats_lookup=stats_lookup,
                matchups_lookup=matchups_lookup,
                retry_delay_seconds=retry_delay_seconds,
                values_lookup=values_lookup,
            )
            stage = "sending the report to Discord"
            send_report(webhook, f"**{name} — {mode.title()} report**\n{report}")
            logger.info("Sent {} report {}/{}", mode, number, len(chosen))
        except Exception:
            failures.append(league_id)
            logger.exception(
                "Failed {} report for league {} ({}) during {}",
                mode,
                name,
                league_id,
                stage,
            )
            if webhook and not args.preview_context:
                try:
                    send_report(
                        webhook,
                        f"Fantasy report failed for {name} ({mode}) while {stage}. "
                        "Check the GitHub Actions run.",
                    )
                except Exception:
                    logger.exception(
                        "Could not send failure notification for league {} ({})",
                        name,
                        league_id,
                    )
    if failures:
        raise RuntimeError(
            f"Reports failed for {len(failures)} of {len(chosen)} leagues: "
            f"{', '.join(failures)}"
        )
    return 0


def main() -> None:
    """Run the command-line program and report failures on stderr."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", backtrace=False, diagnose=False)
    try:
        config = load_config(Path(".env"))
        log_level = (config.get("LOG_LEVEL") or "INFO").upper()
        if log_level != "INFO":
            logger.remove()
            logger.add(sys.stderr, level=log_level, backtrace=False, diagnose=False)
        raise SystemExit(run(config))
    except Exception:
        logger.exception("Fantasy report failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
