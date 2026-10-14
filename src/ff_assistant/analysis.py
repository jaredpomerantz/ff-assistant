"""Model instructions grounded in the structured Sleeper snapshot."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, cast

from google import genai
from google.genai.interactions import Interaction


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


def analyze(task: str, context: dict[str, Any], api_key: str, model: str) -> str:
    """Generate a report from the supplied league snapshot.

    Args:
        task: Report mode to generate.
        context: League and roster data supplied to the model.
        api_key: Gemini API key read from the dotenv file.
        model: Model name read from the dotenv file or its default.

    Returns:
        The generated report text.

    Raises:
        ValueError: The report mode is unknown.
        RuntimeError: The model returns no report text.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    with genai.Client(api_key=api_key) as client:
        response = client.interactions.create(
            model=model,
            system_instruction=SYSTEM,
            input=(
                f"Report mode: {task}\n"
                f"Task instructions: {TASKS[task]}\n"
                f"Data:\n{json.dumps(context, separators=(',', ':'))}"
            ),
            generation_config={"max_output_tokens": 1800, "thinking_level": "low"},
            store=False,
            stream=False,
        )
    report = (cast(Interaction, response).output_text or "").strip()
    if not report:
        raise RuntimeError("Model returned an empty report")
    return report
