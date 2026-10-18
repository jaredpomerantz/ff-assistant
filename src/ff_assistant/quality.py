"""Review a fantasy report for internal contradictions with a second model."""

from __future__ import annotations

from importlib.resources import files
from math import isfinite
from time import sleep

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError
from loguru import logger

from ff_assistant.request_throttle import ModelRequestThrottle

DEFAULT_QA_MODEL = "gemini-3.5-flash-lite"
MAX_QA_ATTEMPTS = 3
REVIEW_INSTRUCTIONS = (
    files("ff_assistant")
    .joinpath("prompts", "quality.txt")
    .read_text(encoding="utf-8")
    .strip()
)


class ReportReviewer:
    """Use a LangChain model call to identify internal contradictions."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_QA_MODEL,
        retry_delay_seconds: float = 5.0,
        request_throttle: ModelRequestThrottle | None = None,
    ) -> None:
        """Create a reusable review chain using the existing Gemini API key."""
        if not isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be a nonnegative finite number")
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REVIEW_INSTRUCTIONS),
                ("human", "Report to review:\n<report>\n{report}\n</report>"),
            ]
        )
        llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,  # type: ignore[arg-type]
            vertexai=False,
            temperature=0,
            max_tokens=2048,
            retries=1,
            request_timeout=120,
        )
        self.model = model
        self.retry_delay_seconds = retry_delay_seconds
        self.request_throttle = request_throttle
        self._chain = prompt | llm | parser
        self._format_instructions = parser.get_format_instructions()

    def review(self, report: str) -> tuple[str, ...]:
        """Return definite contradictions or raise when QA cannot validate a draft."""
        logger.info("Reviewing report consistency with {}", self.model)
        result = None
        for attempt in range(1, MAX_QA_ATTEMPTS + 1):
            if self.request_throttle is not None:
                self.request_throttle.wait(self.model)
            try:
                try:
                    result = self._chain.invoke(
                        {
                            "report": report,
                            "format_instructions": self._format_instructions,
                        }
                    )
                finally:
                    if self.request_throttle is not None:
                        self.request_throttle.complete(self.model)
                break
            except (
                GoogleRateLimitError,
                GoogleAPIError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                if attempt == MAX_QA_ATTEMPTS:
                    logger.error(
                        "QA review attempt {}/{} failed: error_type={}",
                        attempt,
                        MAX_QA_ATTEMPTS,
                        type(exc).__name__,
                    )
                    raise RuntimeError(
                        f"QA review failed ({type(exc).__name__})"
                    ) from None
                logger.warning(
                    "QA review attempt {}/{} failed: error_type={}; "
                    "retrying in {} seconds",
                    attempt,
                    MAX_QA_ATTEMPTS,
                    type(exc).__name__,
                    self.retry_delay_seconds,
                )
                sleep(self.retry_delay_seconds)
            except Exception as exc:
                logger.error(
                    "QA review request failed: error_type={}", type(exc).__name__
                )
                raise RuntimeError(f"QA review failed ({type(exc).__name__})") from None
        if not isinstance(result, dict):
            raise RuntimeError("QA review returned an invalid result")
        consistent = result.get("consistent")
        contradictions = result.get("contradictions")
        if (
            not isinstance(consistent, bool)
            or not isinstance(contradictions, list)
            or len(contradictions) > 5
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 500
                for item in contradictions
            )
            or consistent == bool(contradictions)
        ):
            raise RuntimeError("QA review returned an invalid result")
        logger.info("QA review found {} contradictions", len(contradictions))
        logger.debug("QA contradiction details: {}", contradictions)
        return tuple(contradictions)
