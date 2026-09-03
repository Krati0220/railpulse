"""Root-cause classification: the one place a language model earns its keep.

The engine already maps a known failure code to a :class:`FailureClass` with a
lookup table, and a lookup table is the right tool for that job -- it is
auditable, free, and cannot hallucinate. What it cannot do is read
``"acct bal low - declined"`` or ``"bank server not responding"``, and roughly
a third of real acquirer traffic arrives with no code at all, only prose. Every
one of those classifies as UNKNOWN today.

So the division of labour is deliberate and narrow:

* A code the engine recognises is used as-is. No model call, no cost, no risk.
* Only unrecognised or absent codes reach the model, and all it may do is
  normalise free text into one of a closed set of canonical codes.
* The model never chooses an action, a retry time, a rail, or an amount.
  It proposes a label; deterministic policy decides what that label means.

Three things constrain what comes back:

1. **Structured output.** The provider is forced to emit one member of a
   closed enum plus a confidence. Prose, a novel category, or an injected
   instruction fails validation and is discarded.
2. **A confidence floor.** Below it the answer is thrown away and the case
   goes to MANUAL_REVIEW. Guessing is worse than admitting ignorance when the
   consequence is a retry against a dead instrument.
3. **An audit record.** Every classification stores what went in, what the
   model said, and whether policy accepted it -- so a rejected answer is as
   visible as an accepted one.

Caching is per-message, so a 500-case batch over ~40 distinct issuer strings
costs ~40 calls rather than 500, and a replay of the same batch is free and
deterministic.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 15.0
DEFAULT_MIN_CONFIDENCE = 0.60


class CanonicalCode(StrEnum):
    """The closed vocabulary the deterministic engine understands.

    A model answer outside this set is not a new category to learn from -- it
    is a malformed answer, and it is dropped.
    """

    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    ISSUER_UNAVAILABLE = "ISSUER_UNAVAILABLE"
    GATEWAY_ERROR = "GATEWAY_ERROR"
    CARD_EXPIRED = "CARD_EXPIRED"
    ACCOUNT_CLOSED = "ACCOUNT_CLOSED"
    INVALID_VPA = "INVALID_VPA"
    MANDATE_CANCELLED = "MANDATE_CANCELLED"
    DO_NOT_HONOUR = "DO_NOT_HONOUR"
    SUSPECTED_FRAUD = "SUSPECTED_FRAUD"
    CHARGEBACK_OPEN = "CHARGEBACK_OPEN"


#: Sent to the provider as the schema description. Kept short on purpose: a
#: long prompt is a large surface for the model to be creative on.
CODE_GUIDE = """\
INSUFFICIENT_FUNDS - customer had no money; may succeed later
ISSUER_UNAVAILABLE - the bank or network was down; transient
GATEWAY_ERROR - technical/processing/timeout error; transient
CARD_EXPIRED - card past expiry; needs a new instrument
ACCOUNT_CLOSED - account no longer exists; needs a new instrument
INVALID_VPA - UPI address invalid; needs a new instrument
MANDATE_CANCELLED - the standing mandate was revoked; needs re-authorisation
DO_NOT_HONOUR - generic issuer refusal with no reason given
SUSPECTED_FRAUD - risk or fraud block; do not retry, do not contact
CHARGEBACK_OPEN - a dispute is open; do not retry, do not contact"""

SYSTEM_PROMPT = (
    "You normalise payment failure messages from Indian acquirers into one "
    "canonical code. You are given possibly-messy free text and sometimes a "
    "raw code. Choose exactly one code from the list and give a calibrated "
    "confidence between 0 and 1. If the text is ambiguous or you cannot tell, "
    "give a low confidence rather than guessing confidently -- a wrong "
    "confident answer causes a real retry against a real customer. Keep "
    "the rationale under 15 words.\n\n"
    f"Codes:\n{CODE_GUIDE}"
)


class Classification(BaseModel):
    """The only shape a provider is permitted to return.

    Strictness is deliberately uneven, and the asymmetry is the point.

    ``code`` and ``confidence`` decide whether money moves, so they are rigid:
    a code outside the enum or a confidence outside [0, 1] invalidates the
    whole answer. ``rationale`` is a note for a human reading the audit trail
    and has no effect on any decision, so it is repaired rather than enforced.

    This was originally ``max_length=200`` with no truncation, which meant a
    perfectly correct classification was discarded because the model's
    explanation ran to 227 characters. Against a live provider that rejected
    22 of 24 answers and dropped classifier accuracy to 8.3% -- a guardrail
    strict about prose and, in effect, useless about correctness.
    """

    model_config = ConfigDict(extra="forbid")

    code: CanonicalCode
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("rationale", mode="before")
    @classmethod
    def _truncate_rationale(cls, value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return text if len(text) <= 200 else text[:197] + "..."


@dataclass
class ClassificationRecord:
    """What went in, what came back, and whether policy took it."""

    failure_code: str | None
    issuer_message: str
    source: str  # "lookup" | "model" | "rejected" | "low_confidence" | "error"
    accepted_code: str | None
    confidence: float | None = None
    rationale: str = ""
    reason: str = ""


class LLMProvider(Protocol):
    name: str

    def classify(self, failure_code: str | None, issuer_message: str) -> Classification: ...


# ------------------------------------------------------------------ providers


def _post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


#: Every property is listed in ``required``. OpenAI's strict structured-output
#: mode rejects a schema where a declared property is optional, so leaving
#: ``rationale`` out of ``required`` makes the API refuse the request outright.
#: Anthropic's tool input_schema is happy either way, so one schema serves
#: both. ``Classification.rationale`` still defaults to "", which keeps
#: validation working if a provider omits it anyway.
_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "enum": [c.value for c in CanonicalCode]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "description": "At most 15 words."},
    },
    "required": ["code", "confidence", "rationale"],
    "additionalProperties": False,
}


class AnthropicProvider:
    """Forces a tool call, so the model cannot answer in prose."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5", base_url: str | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")

    def classify(self, failure_code: str | None, issuer_message: str) -> Classification:
        body = _post_json(
            f"{self.base_url}/v1/messages",
            {
                "model": self.model,
                "max_tokens": 256,
                "system": SYSTEM_PROMPT,
                "tools": [
                    {
                        "name": "record_classification",
                        "description": "Record the canonical failure code.",
                        "input_schema": _SCHEMA,
                    }
                ],
                "tool_choice": {"type": "tool", "name": "record_classification"},
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"raw_code: {failure_code or '(none)'}\n"
                            f"issuer_message: {issuer_message}"
                        ),
                    }
                ],
            },
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        for block in body.get("content", []):
            if block.get("type") == "tool_use":
                return Classification.model_validate(block.get("input", {}))
        raise ValueError("provider returned no tool_use block")


class OpenAIProvider:
    """Uses strict JSON-schema structured output."""

    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini", base_url: str | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com").rstrip("/")

    def classify(self, failure_code: str | None, issuer_message: str) -> Classification:
        body = _post_json(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"raw_code: {failure_code or '(none)'}\n"
                            f"issuer_message: {issuer_message}"
                        ),
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "classification",
                        "strict": True,
                        "schema": _SCHEMA,
                    },
                },
            },
            {"authorization": f"Bearer {self.api_key}"},
        )
        content = body["choices"][0]["message"]["content"]
        return Classification.model_validate(json.loads(content))


class KeywordProvider:
    """Offline stand-in so CI, tests and a no-key demo still run.

    This is NOT the AI story -- it is the floor the model has to beat, and the
    evaluation harness reports both so the difference is visible rather than
    claimed.
    """

    name = "keyword"

    _RULES: tuple[tuple[tuple[str, ...], CanonicalCode, float], ...] = (
        (
            ("insufficient", "low balance", "bal low", "nsf", "51"),
            CanonicalCode.INSUFFICIENT_FUNDS,
            0.9,
        ),
        (
            ("issuer", "bank server", "npci", "inoperative", "91", "unavailable"),
            CanonicalCode.ISSUER_UNAVAILABLE,
            0.85,
        ),
        (
            ("timeout", "malfunction", "processing error", "gateway", "96"),
            CanonicalCode.GATEWAY_ERROR,
            0.8,
        ),
        (("expired", "54"), CanonicalCode.CARD_EXPIRED, 0.9),
        (("account closed", "no longer valid"), CanonicalCode.ACCOUNT_CLOSED, 0.85),
        (("vpa",), CanonicalCode.INVALID_VPA, 0.9),
        (("mandate", "si cancelled", "autopay"), CanonicalCode.MANDATE_CANCELLED, 0.85),
        (("fraud", "risk"), CanonicalCode.SUSPECTED_FRAUD, 0.85),
        (("chargeback", "dispute"), CanonicalCode.CHARGEBACK_OPEN, 0.9),
        (
            ("do not honor", "do_not_honour", "not permitted", "no reason", "05"),
            CanonicalCode.DO_NOT_HONOUR,
            0.7,
        ),
    )

    def classify(self, failure_code: str | None, issuer_message: str) -> Classification:
        haystack = f"{failure_code or ''} {issuer_message}".lower()
        for needles, code, confidence in self._RULES:
            if any(needle in haystack for needle in needles):
                return Classification(code=code, confidence=confidence, rationale="keyword match")
        return Classification(code=CanonicalCode.DO_NOT_HONOUR, confidence=0.1, rationale="no rule matched")


# ---------------------------------------------------------------- classifier


@dataclass
class RootCauseClassifier:
    """Lookup first, model only for the gap, policy always last."""

    provider: LLMProvider | None = None
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    #: Pass a dict shared between classifiers to reuse answers across a whole
    #: report. The full run classifies the same ~40 distinct issuer strings in
    #: several sections; without sharing that is ~200 sequential HTTP calls and
    #: about ten minutes of silence. Counters stay per-instance, so each
    #: section still reports its own stats.
    cache: dict[tuple[str | None, str], str | None] = field(default_factory=dict, repr=False)
    records: list[ClassificationRecord] = field(default_factory=list, repr=False)

    #: Counters the writeup quotes.
    lookup_hits: int = 0
    #: Answers served from a cache another section already paid for. Reported
    #: separately so a section showing ``model_calls: 0`` cannot be misread as
    #: the model not having been used.
    cache_hits: int = 0
    model_calls: int = 0
    model_accepted: int = 0
    model_rejected: int = 0
    low_confidence: int = 0
    provider_errors: int = 0

    def normalise(self, failure_code: str | None, issuer_message: str) -> str | None:
        known = (failure_code or "").upper()
        if known in {code.value for code in CanonicalCode}:
            self.lookup_hits += 1
            self._record(failure_code, issuer_message, "lookup", known)
            return known

        if self.provider is None:
            return None

        key = (failure_code, issuer_message)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        self.model_calls += 1
        try:
            result = self.provider.classify(failure_code, issuer_message)
        except ValidationError as exc:
            # The model answered outside the schema. Not a category to learn
            # from -- a malformed answer, dropped.
            self.model_rejected += 1
            # Record the actual field and message, not just a count. "schema:
            # 1 errors" is what made a mass rejection of valid answers hard to
            # diagnose; "rationale: String should have at most 200 characters"
            # would have pointed straight at it.
            detail = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                for err in exc.errors()[:3]
            )
            logger.warning("classifier rejected a model answer -- %s", detail)
            self._record(failure_code, issuer_message, "rejected", None, reason=detail)
            self.cache[key] = None
            return None
        except (urllib.error.URLError, OSError, KeyError, ValueError, TimeoutError) as exc:
            self.provider_errors += 1
            logger.warning("classifier provider failed: %s", exc)
            self._record(failure_code, issuer_message, "error", None, reason=type(exc).__name__)
            # Deliberately not cached: a transport failure is not a verdict.
            return None

        if result.confidence < self.min_confidence:
            self.low_confidence += 1
            self._record(
                failure_code,
                issuer_message,
                "low_confidence",
                None,
                confidence=result.confidence,
                rationale=result.rationale,
                reason=f"below floor {self.min_confidence}",
            )
            self.cache[key] = None
            return None

        self.model_accepted += 1
        self._record(
            failure_code,
            issuer_message,
            "model",
            result.code.value,
            confidence=result.confidence,
            rationale=result.rationale,
        )
        self.cache[key] = result.code.value
        return result.code.value

    def _record(
        self,
        failure_code: str | None,
        issuer_message: str,
        source: str,
        accepted: str | None,
        confidence: float | None = None,
        rationale: str = "",
        reason: str = "",
    ) -> None:
        self.records.append(
            ClassificationRecord(
                failure_code=failure_code,
                issuer_message=issuer_message,
                source=source,
                accepted_code=accepted,
                confidence=confidence,
                rationale=rationale,
                reason=reason,
            )
        )

    def stats(self) -> dict[str, int | float]:
        return {
            "lookup_hits": self.lookup_hits,
            "cache_hits": self.cache_hits,
            "model_calls": self.model_calls,
            "model_accepted": self.model_accepted,
            "model_rejected": self.model_rejected,
            "low_confidence": self.low_confidence,
            "provider_errors": self.provider_errors,
            "acceptance_rate": (
                self.model_accepted / self.model_calls if self.model_calls else 0.0
            ),
        }


def build_provider(
    anthropic_key: str | None = None,
    openai_key: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """Anthropic, then OpenAI, then the offline keyword floor.

    ``model`` overrides the per-provider default; ``RAILPULSE_MODEL`` does the
    same from the environment, so a run can be re-pointed at a different model
    without touching code. Setting only ``OPENAI_API_KEY`` selects OpenAI --
    Anthropic wins only when both keys are present.
    """
    model = model or os.getenv("RAILPULSE_MODEL")
    if anthropic_key:
        return AnthropicProvider(anthropic_key, model or "claude-sonnet-4-5")
    if openai_key:
        return OpenAIProvider(openai_key, model or "gpt-4.1-mini")
    return KeywordProvider()
