"""Discord webhook delivery with message length and mention safeguards."""

from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger


def split_report(report: str, limit: int = 1900) -> list[str]:
    """Split a report into Discord-sized messages.

    Args:
        report: Text to split.
        limit: Maximum number of characters in each message.

    Returns:
        The report chunks in order.
    """
    chunks = []
    remaining = report.strip()
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        end = remaining.rfind("\n", 0, limit)
        if end < limit // 2:
            end = limit
        chunks.append(remaining[:end].rstrip())
        remaining = remaining[end:].lstrip("\n")
    return chunks


def send_report(webhook_url: str, report: str) -> None:
    """Post report chunks to a Discord webhook without mentions.

    Args:
        webhook_url: Destination webhook URL.
        report: Report text to send.

    Raises:
        RuntimeError: Delivery fails after the retry limit.
    """
    chunks = split_report(report)
    for number, chunk in enumerate(chunks, start=1):
        body = json.dumps(
            {"content": chunk, "allowed_mentions": {"parse": []}}
        ).encode()
        request = Request(
            webhook_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ff-assistant/0.1",
            },
            method="POST",
        )
        for attempt in range(3):
            try:
                with urlopen(request, timeout=25):
                    logger.info("Sent Discord chunk {}/{}", number, len(chunks))
                    break
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(
                        f"Discord delivery failed (HTTP {exc.code})"
                    ) from None
                delay = min(float(exc.headers.get("Retry-After", "2")), 10)
                logger.warning(
                    "Discord returned HTTP {}; retrying chunk {}/{} in {} seconds",
                    exc.code,
                    number,
                    len(chunks),
                    delay,
                )
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise RuntimeError(
                        "Discord delivery failed (network error)"
                    ) from None
                delay = 2 * (attempt + 1)
                logger.warning(
                    "Discord network error; retrying chunk {}/{} in {} seconds",
                    number,
                    len(chunks),
                    delay,
                )
            time.sleep(delay)
