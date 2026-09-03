"""Guarded AI outreach-preview boundary.

This module intentionally produces a *preview*, never a delivery request. A
provider may draft structured copy, but deterministic policy owns the URL,
financial promises, and whether the draft is safe to show a merchant.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.classifier import _post_json
from app.models import RecoveryCase

logger = logging.getLogger(__name__)

Language = Literal["en", "hi", "hinglish"]
Tone = Literal["polite_urgent", "polite", "reassuring"]


class OutreachContext(BaseModel):
    """The least-privilege context a future Instructor/LLM provider receives."""

    model_config = ConfigDict(extra="forbid")

    language: Language
    tone: Tone = "polite_urgent"
    recovery_reason: str
    rail: str


class OutreachDraft(BaseModel):
    """The only structured shape an LLM/provider is allowed to return."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    language: Language
    tone: Tone
    message_body: str = Field(min_length=20, max_length=320)


class OutreachPreview(BaseModel):
    """Merchant-facing result of a policy validation; delivery is always off."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    language: Language
    tone: Tone
    message_body: str | None = None
    final_message: str | None = None
    blocked_reasons: list[str] = Field(default_factory=list)
    policy_checks: list[str] = Field(default_factory=list)
    #: Which provider wrote this. "template" means three hardcoded strings, not
    #: a model -- the same distinction the report's NO API KEY SET banner makes,
    #: surfaced here so an API consumer or a dashboard cannot present a lookup
    #: as AI without saying so.
    drafted_by: str = "template"
    preview_only: Literal[True] = True


class DraftProvider(Protocol):
    def draft(self, context: OutreachContext) -> OutreachDraft | dict[str, str]: ...


class TemplateDraftProvider:
    """Three hardcoded strings. The floor a model has to beat, not the AI."""

    name = "template"

    _TEMPLATES = {
        "en": "Hi, we could not complete your subscription renewal. Please use the secure payment link below to keep your service active.",
        "hi": "नमस्ते, आपका सब्सक्रिप्शन रिन्यूअल पूरा नहीं हो पाया। अपनी सेवा जारी रखने के लिए नीचे दिए सुरक्षित पेमेंट लिंक का उपयोग करें।",
        "hinglish": "Hi, aapka subscription renewal complete nahi ho paaya. Service active rakhne ke liye neeche diye secure payment link se pay karein.",
    }

    def draft(self, context: OutreachContext) -> OutreachDraft:
        return OutreachDraft(
            language=context.language,
            tone=context.tone,
            message_body=self._TEMPLATES[context.language],
        )


#: What the model is allowed to return. Deliberately the same shape as
#: ``OutreachDraft`` and nothing more: no URL field, no amount field, no
#: recipient field. The model cannot supply a link because there is nowhere to
#: put one, which is a stronger guarantee than asking it not to.
_DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "enum": ["en", "hi", "hinglish"]},
        "tone": {"type": "string", "enum": ["polite_urgent", "polite", "reassuring"]},
        "message_body": {
            "type": "string",
            "description": "20-320 characters. No URLs, no discounts, no threats.",
        },
    },
    "required": ["language", "tone", "message_body"],
    "additionalProperties": False,
}

DRAFT_SYSTEM_PROMPT = """You draft one short message asking a customer to \
complete a failed subscription payment.

Hard rules:
- Never include a URL, link, domain or phone number. One is appended by code.
- Never offer a discount, refund, waiver, coupon, cashback or anything free.
- Never threaten: no legal action, no account suspension, no "final warning".
- Write in the requested language. "hinglish" means romanised Hindi-English.
- 20 to 320 characters. One message, no alternatives, no commentary.

You are drafting for a merchant's review, not sending. A draft breaking any \
rule above is discarded by a validator you cannot influence, so there is no \
advantage in trying."""


class LLMDraftProvider:
    """A real model behind the drafting seam.

    The seam has existed since the first commit and nothing ever filled it --
    ``OutreachCopilot`` defaulted to three hardcoded strings, and the README
    described "the AI copilot" as though a model were writing them. It was not.
    The guardrails below were real and tested; the thing they guarded was a
    dictionary lookup.

    What makes this safe to add is that none of the safety lives here. The
    schema has no field for a URL or an amount, so a model cannot supply one --
    a structural guarantee rather than an instruction it might ignore. Whatever
    it does return still passes through ``OutreachPolicyValidator``, and
    delivery is still off. This provider can be prompt-injected, jailbroken or
    simply wrong, and the worst outcome is a draft a merchant is shown and
    declines.
    """

    def __init__(
        self,
        api_key: str,
        *,
        vendor: Literal["anthropic", "openai"],
        model: str,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.vendor = vendor
        self.model = model
        # Carries the model name, so a preview records which one wrote it and
        # a stale draft can be traced to the version that produced it.
        self.name = f"llm:{vendor}:{model}"
        default = (
            "https://api.anthropic.com" if vendor == "anthropic" else "https://api.openai.com"
        )
        self.base_url = (base_url or default).rstrip("/")

    def draft(self, context: OutreachContext) -> OutreachDraft:
        instruction = (
            f"language: {context.language}\n"
            f"tone: {context.tone}\n"
            f"why the payment failed: {context.recovery_reason}\n"
            f"payment method: {context.rail}"
        )
        payload = (
            self._anthropic(instruction)
            if self.vendor == "anthropic"
            else self._openai(instruction)
        )
        # Validated here so a malformed answer raises before it reaches the
        # copilot, which treats a provider failure as "fall back to template".
        return OutreachDraft.model_validate(payload)

    def _anthropic(self, instruction: str) -> dict:
        body = _post_json(
            f"{self.base_url}/v1/messages",
            {
                "model": self.model,
                "max_tokens": 512,
                "system": DRAFT_SYSTEM_PROMPT,
                "tools": [
                    {
                        "name": "record_draft",
                        "description": "Record the drafted message.",
                        "input_schema": _DRAFT_SCHEMA,
                    }
                ],
                # Forced, so the model cannot answer in prose and cannot
                # decline to use the schema.
                "tool_choice": {"type": "tool", "name": "record_draft"},
                "messages": [{"role": "user", "content": instruction}],
            },
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
        )
        for block in body.get("content", []):
            if block.get("type") == "tool_use":
                return block["input"]
        raise ValueError("anthropic returned no tool_use block")

    def _openai(self, instruction: str) -> dict:
        body = _post_json(
            f"{self.base_url}/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "outreach_draft",
                        "strict": True,
                        "schema": _DRAFT_SCHEMA,
                    },
                },
            },
            {"authorization": f"Bearer {self.api_key}"},
        )
        return json.loads(body["choices"][0]["message"]["content"])


def build_draft_provider(
    anthropic_key: str | None = None,
    openai_key: str | None = None,
    model: str | None = None,
) -> DraftProvider:
    """A model if a key is configured, the template floor otherwise.

    Mirrors ``classifier.build_provider`` deliberately: one key, both AI
    surfaces, and the same rule about which vendor wins.
    """
    model = model or os.getenv("RAILPULSE_MODEL")
    if anthropic_key:
        return LLMDraftProvider(
            anthropic_key, vendor="anthropic", model=model or "claude-sonnet-4-5"
        )
    if openai_key:
        return LLMDraftProvider(openai_key, vendor="openai", model=model or "gpt-4.1-mini")
    return TemplateDraftProvider()


class NonCompliantDraftProvider:
    """Returns exactly what a model must never be trusted to send.

    Every guardrail in this module was unreachable in the demo: the template
    provider only ever emits three hardcoded, compliant strings, so the
    validator could never reject anything and the dashboard's "blocked" branch
    was dead code. A rule nobody has watched refuse something is a claim, not a
    demonstration.

    This draft trips all three checks at once -- a model-supplied URL, an
    invented discount, and coercive dunning language.
    """

    name = "non_compliant_demo"

    def draft(self, context: OutreachContext) -> OutreachDraft:
        return OutreachDraft(
            language=context.language,
            tone=context.tone,
            message_body=(
                "Final warning: pay immediately or your account will be blocked. "
                "Get 20% off if you settle today at www.quick-pay-now.example"
            ),
        )


class OutreachPolicyValidator:
    """Rules that cannot be bypassed by a prompt or model output."""

    _url_pattern = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
    _offer_pattern = re.compile(
        r"(?i)\b(?:\d{1,3}%\s*(?:off|discount|cashback)|discount|cashback|coupon|offer|waiv(?:e|ed|er)|free)\b"
    )
    _coercive_pattern = re.compile(
        r"(?i)\b(?:legal action|final warning|account (?:will be )?(?:blocked|suspended)|pay immediately|pay now or)\b"
    )

    @classmethod
    def validate(cls, draft: OutreachDraft, payment_link_url: str) -> list[str]:
        reasons: list[str] = []
        body = draft.message_body
        if cls._url_pattern.search(body):
            reasons.append("model_supplied_url")
        if cls._offer_pattern.search(body):
            reasons.append("unapproved_offer_or_discount")
        if cls._coercive_pattern.search(body):
            reasons.append("coercive_dunning_language")
        if not cls._is_canonical_payment_link(payment_link_url):
            reasons.append("invalid_canonical_payment_link")
        return reasons

    @staticmethod
    def _is_canonical_payment_link(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.netloc.lower() == "rzp.io" and bool(parsed.path)


class OutreachCopilot:
    """Structured drafting plus deterministic merchant-risk guardrails."""

    def __init__(self, provider: DraftProvider | None = None) -> None:
        self.provider = provider or TemplateDraftProvider()

    def generate_preview(
        self,
        case: RecoveryCase,
        payment_link_url: str,
        *,
        language: Language = "hinglish",
    ) -> OutreachPreview:
        context = OutreachContext(
            language=language,
            recovery_reason=case.failure_class.value if case.failure_class else "unknown",
            rail=case.rail.value,
        )
        try:
            draft = OutreachDraft.model_validate(self.provider.draft(context))
        except ValidationError:
            return OutreachPreview(
                approved=False,
                language=language,
                tone="polite_urgent",
                blocked_reasons=["invalid_structured_draft"],
                policy_checks=["structured_output_required"],
                drafted_by=self._provider_name,
            )
        except Exception:
            # A live provider can time out, rate-limit or return nonsense. That
            # must degrade to the template rather than fail the preview: the
            # merchant asked to see a draft, not to hear about our vendor.
            logger.warning("draft provider failed; falling back to template", exc_info=True)
            draft = TemplateDraftProvider().draft(context)
            return self._approve(draft, payment_link_url, drafted_by="template_after_provider_error")

        return self._approve(draft, payment_link_url, drafted_by=self._provider_name)

    @property
    def _provider_name(self) -> str:
        return getattr(self.provider, "name", type(self.provider).__name__)

    def _approve(
        self, draft: OutreachDraft, payment_link_url: str, *, drafted_by: str
    ) -> OutreachPreview:
        blocked_reasons = OutreachPolicyValidator.validate(draft, payment_link_url)
        if blocked_reasons:
            return OutreachPreview(
                approved=False,
                language=draft.language,
                tone=draft.tone,
                message_body=draft.message_body,
                blocked_reasons=blocked_reasons,
                policy_checks=["no_model_urls", "no_unapproved_offers", "no_coercive_language"],
                drafted_by=drafted_by,
            )

        # Only policy-owned code appends the canonical Razorpay link.
        return OutreachPreview(
            approved=True,
            language=draft.language,
            tone=draft.tone,
            message_body=draft.message_body,
            final_message=f"{draft.message_body}\n\nPay securely: {payment_link_url}",
            policy_checks=[
                "structured_output_valid",
                "no_model_urls",
                "no_unapproved_offers",
                "no_coercive_language",
                "canonical_link_appended_by_code",
            ],
            drafted_by=drafted_by,
        )
