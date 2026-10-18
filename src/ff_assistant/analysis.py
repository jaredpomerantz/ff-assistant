"""Model instructions grounded in the structured Sleeper snapshot."""

from __future__ import annotations

import json
from importlib.resources import files
from math import isfinite
from time import monotonic, sleep
from typing import Any, cast

from google import genai
from google.genai import errors
from google.genai.interactions import Interaction
from loguru import logger

from ff_assistant.player_matchups import PlayerMatchupLookup
from ff_assistant.player_stats import PlayerStatsLookup
from ff_assistant.player_values import ATTRIBUTION, PlayerValuesLookup


def _read_prompt(name: str) -> str:
    """Read a prompt bundled with the installed package."""
    return (
        files("ff_assistant")
        .joinpath("prompts", name)
        .read_text(encoding="utf-8")
        .strip()
    )


SYSTEM = _read_prompt("system.txt")
TASKS = {
    task: _read_prompt(f"{task}.txt") for task in ("waivers", "injuries", "lineup")
}
MAX_TOOL_ROUNDS = 4
LINEUP_MAX_TOOL_ROUNDS = 8
MAX_OUTPUT_TOKENS = 1800
LINEUP_MAX_OUTPUT_TOKENS = 8192
MAX_GEMINI_ATTEMPTS = 3
PLAYER_STATS_TOOL = {
    "type": "function",
    "name": "get_player_stats",
    "description": (
        "Look up regular-season counting, volume, efficiency, and available "
        "offensive snap stats for Sleeper player IDs in the supplied report data. "
        "Returns season totals and a recent-games summary."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "player_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One to twelve Sleeper player IDs from the report data.",
            },
            "recent_games": {
                "type": "integer",
                "description": "Number of most recent played games, from one to five.",
            },
        },
        "required": ["player_ids"],
    },
}
PLAYER_MATCHUPS_TOOL = {
    "type": "function",
    "name": "get_player_matchups",
    "description": (
        "Look up current-week NFL opponents, byes, and kickoff times for Sleeper "
        "player IDs in the supplied report data. Does not project fantasy points."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "player_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One to twelve Sleeper player IDs from the report data.",
            },
        },
        "required": ["player_ids"],
    },
}
PLAYER_VALUES_TOOL = {
    "type": "function",
    "name": "get_player_values",
    "description": (
        "Look up current TheTradeCalc dynasty trade values and overall/position "
        "rankings for Sleeper player IDs in this report. Values use a 12-team "
        "Superflex PPR baseline and are estimates, not projections."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "player_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One to twelve Sleeper player IDs from the report data.",
            },
        },
        "required": ["player_ids"],
    },
}


def _allowed_player_ids(context: dict[str, Any]) -> set[str]:
    return {
        str(player["id"])
        for player in (
            context.get("my_roster", {}).get("players", [])
            + context.get("available_trending_adds", [])
        )
        if player.get("id") is not None
    }


def _run_stats_tool(
    arguments: Any, context: dict[str, Any], stats_lookup: PlayerStatsLookup
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {"status": "invalid_request", "reason": "Invalid tool arguments"}
    player_ids = arguments.get("player_ids")
    if (
        not isinstance(player_ids, list)
        or any(not isinstance(player_id, str) for player_id in player_ids)
        or not set(player_ids) <= _allowed_player_ids(context)
    ):
        return {
            "status": "invalid_request",
            "reason": "Player ID is not in this report",
        }
    return stats_lookup.lookup(player_ids, arguments.get("recent_games", 3))


def _run_matchups_tool(
    arguments: Any, context: dict[str, Any], matchups_lookup: PlayerMatchupLookup
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {"status": "invalid_request", "reason": "Invalid tool arguments"}
    player_ids = arguments.get("player_ids")
    if (
        not isinstance(player_ids, list)
        or any(not isinstance(player_id, str) for player_id in player_ids)
        or not set(player_ids) <= _allowed_player_ids(context)
    ):
        return {
            "status": "invalid_request",
            "reason": "Player ID is not in this report",
        }
    return matchups_lookup.lookup(player_ids)


def _run_values_tool(
    arguments: Any, context: dict[str, Any], values_lookup: PlayerValuesLookup
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {"status": "invalid_request", "reason": "Invalid tool arguments"}
    player_ids = arguments.get("player_ids")
    if (
        not isinstance(player_ids, list)
        or any(not isinstance(player_id, str) for player_id in player_ids)
        or not set(player_ids) <= _allowed_player_ids(context)
    ):
        return {
            "status": "invalid_request",
            "reason": "Player ID is not in this report",
        }
    return values_lookup.lookup(player_ids)


def _request_turn(
    client: genai.Client,
    model: str,
    history: list[dict[str, Any]],
    turn_number: int,
    retry_delay_seconds: float,
    max_output_tokens: int,
) -> tuple[Interaction, float, int]:
    """Retry transient Gemini request failures with the configured delay."""
    started = monotonic()
    for attempt in range(1, MAX_GEMINI_ATTEMPTS + 1):
        try:
            response = cast(
                Interaction,
                client.interactions.create(
                    model=model,
                    system_instruction=SYSTEM,
                    input=history,
                    tools=[PLAYER_STATS_TOOL, PLAYER_MATCHUPS_TOOL, PLAYER_VALUES_TOOL],
                    generation_config={"max_output_tokens": max_output_tokens},
                    store=False,
                    stream=False,
                ),
            )
            return response, monotonic() - started, attempt
        except errors.APIError as exc:
            headers = getattr(exc.response, "headers", None)
            request_id = None
            if headers is not None:
                for header in ("x-request-id", "x-goog-request-id"):
                    request_id = headers.get(header)
                    if request_id:
                        break
            retryable = exc.code in (408, 429) or 500 <= exc.code < 600
            retry = retryable and attempt < MAX_GEMINI_ATTEMPTS
            log = logger.warning if retry else logger.error
            log(
                "Gemini turn {} attempt {}/{} failed: model={} error_type={} "
                "http_status={} api_status={} request_id={} elapsed_s={:.2f}",
                turn_number,
                attempt,
                MAX_GEMINI_ATTEMPTS,
                model,
                type(exc).__name__,
                exc.code,
                exc.status,
                request_id,
                monotonic() - started,
            )
            if retry:
                logger.warning(
                    "Retrying Gemini turn {} in {} seconds",
                    turn_number,
                    retry_delay_seconds,
                )
                sleep(retry_delay_seconds)
                continue
            raise RuntimeError(
                f"Gemini turn {turn_number} failed "
                f"(HTTP {exc.code}, status={exc.status}, request_id={request_id})"
            ) from None
        except (TimeoutError, ConnectionError) as exc:
            retry = attempt < MAX_GEMINI_ATTEMPTS
            log = logger.warning if retry else logger.error
            log(
                "Gemini turn {} attempt {}/{} failed: model={} error_type={} "
                "elapsed_s={:.2f}",
                turn_number,
                attempt,
                MAX_GEMINI_ATTEMPTS,
                model,
                type(exc).__name__,
                monotonic() - started,
            )
            if retry:
                logger.warning(
                    "Retrying Gemini turn {} in {} seconds",
                    turn_number,
                    retry_delay_seconds,
                )
                sleep(retry_delay_seconds)
                continue
            raise RuntimeError(
                f"Gemini turn {turn_number} failed ({type(exc).__name__})"
            ) from None
        except Exception as exc:
            logger.error(
                "Gemini turn {} attempt {}/{} failed: model={} error_type={} "
                "elapsed_s={:.2f}",
                turn_number,
                attempt,
                MAX_GEMINI_ATTEMPTS,
                model,
                type(exc).__name__,
                monotonic() - started,
            )
            raise RuntimeError(
                f"Gemini turn {turn_number} failed ({type(exc).__name__})"
            ) from None
    raise RuntimeError("Gemini request exhausted all attempts")


def analyze(
    task: str,
    context: dict[str, Any],
    api_key: str,
    model: str,
    stats_lookup: PlayerStatsLookup,
    matchups_lookup: PlayerMatchupLookup,
    retry_delay_seconds: float = 5.0,
    values_lookup: PlayerValuesLookup | None = None,
) -> str:
    """Generate a report from the supplied league snapshot.

    Args:
        task: Report mode to generate.
        context: League and roster data supplied to the model.
        api_key: Gemini API key read from the dotenv file.
        model: Model name read from the dotenv file or its default.
        stats_lookup: Season-scoped lookup for nflverse player statistics.
        matchups_lookup: Week-scoped lookup for NFL player matchups.
        retry_delay_seconds: Delay between transient Gemini request attempts.
        values_lookup: Optional source for current dynasty values and rankings.

    Returns:
        The generated report text.

    Raises:
        ValueError: The report mode is unknown.
        RuntimeError: The model returns no report text.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if not isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be a nonnegative finite number")
    prompt = (
        f"Report mode: {task}\n"
        f"Task instructions: {TASKS[task]}\n"
        f"Data:\n{json.dumps(context, separators=(',', ':'))}"
    )
    history: list[dict[str, Any]] = [
        {"type": "user_input", "content": [{"type": "text", "text": prompt}]}
    ]
    max_tool_rounds = LINEUP_MAX_TOOL_ROUNDS if task == "lineup" else MAX_TOOL_ROUNDS
    max_output_tokens = (
        LINEUP_MAX_OUTPUT_TOKENS if task == "lineup" else MAX_OUTPUT_TOKENS
    )
    used_values = False
    with genai.Client(api_key=api_key) as client:
        for turn in range(max_tool_rounds + 1):
            turn_number = turn + 1
            logger.info(
                "Requesting Gemini turn {} for {} report: model={} "
                "history_entries={} prompt_chars={} max_output_tokens={}",
                turn_number,
                task,
                model,
                len(history),
                len(prompt),
                max_output_tokens,
            )
            response, elapsed, attempt = _request_turn(
                client,
                model,
                history,
                turn_number,
                retry_delay_seconds,
                max_output_tokens,
            )
            steps = response.steps or []
            calls = [step for step in steps if step.type == "function_call"]
            usage = response.usage
            logger.debug(
                "Gemini turn {} returned: status={} interaction_id={} "
                "attempt={} elapsed_s={:.2f} input_tokens={} output_tokens={} "
                "thought_tokens={} tool_calls={}",
                turn_number,
                response.status,
                response.id,
                attempt,
                elapsed,
                usage.total_input_tokens if usage else None,
                usage.total_output_tokens if usage else None,
                usage.total_thought_tokens if usage else None,
                len(calls),
            )
            logger.debug(
                "Gemini turn {} model output:\n{}",
                turn_number,
                response.output_text if response.output_text is not None else "<none>",
            )
            if not calls:
                report = (response.output_text or "").strip()
                if not report:
                    error_codes = [
                        error.code for error in (response.errors or []) if error.code
                    ]
                    step_types = [step.type for step in steps]
                    input_tokens = usage.total_input_tokens if usage else None
                    output_tokens = usage.total_output_tokens if usage else None
                    thought_tokens = usage.total_thought_tokens if usage else None
                    raise RuntimeError(
                        "Model returned an empty report "
                        f"(turn={turn_number}, status={response.status}, "
                        f"interaction_id={response.id}, error_codes={error_codes}, "
                        f"step_types={step_types}, "
                        f"input_tokens={input_tokens}, "
                        f"output_tokens={output_tokens}, "
                        f"thought_tokens={thought_tokens}, "
                        f"max_output_tokens={max_output_tokens})"
                    )
                if used_values and ATTRIBUTION not in report:
                    separator = "-" * 85
                    if report.endswith(separator):
                        body = report[: -len(separator)].rstrip()
                        report = f"{body}\n{ATTRIBUTION}\n{separator}"
                    else:
                        report = f"{report}\n{ATTRIBUTION}"
                logger.info("Generated {} report with {} characters", task, len(report))
                return report
            if turn == max_tool_rounds:
                raise RuntimeError(
                    f"Model exceeded the tool call limit on turn {turn_number} "
                    f"(max_tool_rounds={max_tool_rounds}, status={response.status}, "
                    f"interaction_id={response.id})"
                )
            history.extend(step.model_dump() for step in steps)
            logger.info(
                "Answering {} model tool calls on turn {}: names={}",
                len(calls),
                turn_number,
                [call.name for call in calls],
            )
            for call in calls:
                if call.name == "get_player_stats":
                    result = _run_stats_tool(call.arguments, context, stats_lookup)
                elif call.name == "get_player_matchups":
                    result = _run_matchups_tool(
                        call.arguments, context, matchups_lookup
                    )
                elif call.name == "get_player_values" and values_lookup is not None:
                    result = _run_values_tool(call.arguments, context, values_lookup)
                    used_values |= result.get("status") == "available" and any(
                        player.get("status") == "available"
                        for player in result.get("players", [])
                    )
                else:
                    result = {"status": "invalid_request", "reason": "Unknown tool"}
                history.append(
                    {
                        "type": "function_result",
                        "name": call.name,
                        "call_id": call.id,
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(result, separators=(",", ":")),
                            }
                        ],
                    }
                )
    raise RuntimeError("Model did not finish the report")
