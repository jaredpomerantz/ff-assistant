"""Space model requests within one fantasy report run."""

from __future__ import annotations

from math import isfinite
from time import monotonic, sleep

from loguru import logger

FLASH_LITE_MODEL = "gemini-3.5-flash-lite"
MIN_FLASH_LITE_INTERVAL_SECONDS = 4.1


class ModelRequestThrottle:
    """Enforce a minimum gap after each request to the same model."""

    def __init__(self, interval_seconds: float) -> None:
        """Set the request interval, with a floor for the 15 RPM model."""
        if not isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError("interval_seconds must be a nonnegative finite number")
        self.interval_seconds = interval_seconds
        self._next_allowed_at: dict[str, float] = {}

    def wait(self, model: str) -> None:
        """Pause until another request to this model can start."""
        key = model.removeprefix("models/")
        remaining = self._next_allowed_at.get(key, 0.0) - monotonic()
        if remaining > 0:
            logger.info(
                "Waiting {:.2f} seconds before next {} request", remaining, model
            )
            sleep(remaining)

    def complete(self, model: str) -> None:
        """Start the quiet period when a request succeeds or fails."""
        key = model.removeprefix("models/")
        interval = self.interval_seconds
        if key == FLASH_LITE_MODEL:
            interval = max(interval, MIN_FLASH_LITE_INTERVAL_SECONDS)
        self._next_allowed_at[key] = monotonic() + interval
