"""Test per-model spacing without sleeping in real time."""

from __future__ import annotations

import pytest

from ff_assistant.request_throttle import ModelRequestThrottle


def test_throttle_spaces_same_model_and_separates_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Share a request gap within one model and allow independent model clocks."""
    now = [100.0]
    waits: list[float] = []

    def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    monkeypatch.setattr("ff_assistant.request_throttle.monotonic", lambda: now[0])
    monkeypatch.setattr("ff_assistant.request_throttle.sleep", fake_sleep)
    throttle = ModelRequestThrottle(5.0)

    throttle.wait("gemini-3.5-flash-lite")
    throttle.complete("gemini-3.5-flash-lite")
    throttle.wait("another-model")
    throttle.complete("another-model")
    throttle.wait("gemini-3.5-flash-lite")
    throttle.complete("gemini-3.5-flash-lite")
    throttle.wait("gemini-3.5-flash-lite")

    assert waits == [5.0, 5.0]


@pytest.mark.parametrize("interval", [-1.0, float("inf"), float("nan")])
def test_throttle_rejects_invalid_interval(interval: float) -> None:
    """Reject settings that cannot produce meaningful request spacing."""
    with pytest.raises(ValueError, match="nonnegative finite"):
        ModelRequestThrottle(interval)


def test_flash_lite_keeps_15_rpm_floor_even_with_shorter_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat prefixed and bare model names as the same 15 RPM quota."""
    now = [100.0]
    waits: list[float] = []

    def fake_sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    monkeypatch.setattr("ff_assistant.request_throttle.monotonic", lambda: now[0])
    monkeypatch.setattr("ff_assistant.request_throttle.sleep", fake_sleep)
    throttle = ModelRequestThrottle(1.0)

    throttle.wait("models/gemini-3.5-flash-lite")
    throttle.complete("models/gemini-3.5-flash-lite")
    throttle.wait("gemini-3.5-flash-lite")

    assert waits == pytest.approx([4.1])
