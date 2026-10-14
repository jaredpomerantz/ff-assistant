"""Read application settings from the repository's .env file."""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values


def load_config(path: Path) -> dict[str, str]:
    """Read settings from a dotenv file without consulting process variables.

    Args:
        path: Path to the dotenv file.

    Returns:
        Defined settings with string values.

    Raises:
        FileNotFoundError: The dotenv file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return {
        key: value
        for key, value in dotenv_values(path, interpolate=False).items()
        if value is not None
    }
