"""Model instructions grounded in the structured Sleeper snapshot."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, cast

from google import genai
from google.genai.interactions import Interaction
from loguru import logger

from ff_assistant.player_stats import PlayerStatsLookup


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


def _run_stats_tool(
    arguments: Any, context: dict[str, Any], stats_lookup: PlayerStatsLookup
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {"status": "invalid_request", "reason": "Invalid tool arguments"}
    player_ids = arguments.get("player_ids")
    allowed_ids = {
        str(player["id"])
        for player in (
            context.get("my_roster", {}).get("players", [])
            + context.get("available_trending_adds", [])
        )
        if player.get("id") is not None
    }
    if (
        not isinstance(player_ids, list)
        or any(not isinstance(player_id, str) for player_id in player_ids)
        or not set(player_ids) <= allowed_ids
    ):
        return {
            "status": "invalid_request",
            "reason": "Player ID is not in this report",
        }
    return stats_lookup.lookup(player_ids, arguments.get("recent_games", 3))


def analyze(
    task: str,
    context: dict[str, Any],
    api_key: str,
    model: str,
    stats_lookup: PlayerStatsLookup,
) -> str:
    """Generate a report from the supplied league snapshot.

    Args:
        task: Report mode to generate.
        context: League and roster data supplied to the model.
        api_key: Gemini API key read from the dotenv file.
        model: Model name read from the dotenv file or its default.
        stats_lookup: Season-scoped lookup for nflverse player statistics.

    Returns:
        The generated report text.

    Raises:
        ValueError: The report mode is unknown.
        RuntimeError: The model returns no report text.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    prompt = (
        f"Report mode: {task}\n"
        f"Task instructions: {TASKS[task]}\n"
        f"Data:\n{json.dumps(context, separators=(',', ':'))}"
    )
    history: list[dict[str, Any]] = [
        {"type": "user_input", "content": [{"type": "text", "text": prompt}]}
    ]
    with genai.Client(api_key=api_key) as client:
        for turn in range(MAX_TOOL_ROUNDS + 1):
            response = cast(
                Interaction,
                client.interactions.create(
                    model=model,
                    system_instruction=SYSTEM,
                    input=history,
                    tools=[PLAYER_STATS_TOOL],
                    generation_config={"max_output_tokens": 1800},
                    store=False,
                    stream=False,
                ),
            )
            steps = response.steps or []
            calls = [step for step in steps if step.type == "function_call"]
            if not calls:
                report = (response.output_text or "").strip()
                if not report:
                    raise RuntimeError("Model returned an empty report")
                logger.info("Generated {} report with {} characters", task, len(report))
                return report
            if turn == MAX_TOOL_ROUNDS:
                raise RuntimeError("Model exceeded the player-stats tool call limit")
            history.extend(step.model_dump() for step in steps)
            logger.info("Answering {} player-stats tool calls", len(calls))
            for call in calls:
                result = (
                    _run_stats_tool(call.arguments, context, stats_lookup)
                    if call.name == "get_player_stats"
                    else {"status": "invalid_request", "reason": "Unknown tool"}
                )
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
