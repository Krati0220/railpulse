"""The second place a model touches this system, and what stops it mattering.

The drafting seam existed from the first commit and nothing ever filled it.
``OutreachCopilot`` defaulted to three hardcoded strings while the README
described "the AI copilot" as though a model were writing them -- the
guardrails were real and tested, and the thing they guarded was a dictionary
lookup. ``LLMDraftProvider`` fills it.

Adding a model to a system that moves money is normally where the risk enters.
Here it should not, and these tests are the argument for why:

* the schema has no field for a URL or an amount, so a model cannot supply one
  regardless of what it is asked -- structure, not instruction;
* whatever it returns still passes the validator, which no prompt can reach;
* a provider that times out or returns nonsense degrades to the template
  instead of failing the merchant's request;
* and the preview says which of those happened, so a lookup can never be
  presented as a model.

Every test below drives a fake transport. Nothing here reaches the network.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

from app.ai_copilot import (
    LLMDraftProvider,
    OutreachContext,
    OutreachCopilot,
    OutreachDraft,
    TemplateDraftProvider,
    build_draft_provider,
)
from app.models import FailureClass, PaymentRail, RecoveryCase

LINK = "https://rzp.io/i/abc123"
CONTEXT = OutreachContext(language="en", recovery_reason="customer_action", rail="card")


def case() -> RecoveryCase:
    return RecoveryCase(
        id="case_1",
        logical_key="inv_1",
        amount_paise=49900,
        failure_class=FailureClass.CUSTOMER_ACTION,
        rail=PaymentRail.CARD,
    )


class ProviderSelectionTests(unittest.TestCase):
    def test_no_key_means_no_model(self) -> None:
        self.assertIsInstance(build_draft_provider(None, None), TemplateDraftProvider)

    def test_a_key_selects_a_model(self) -> None:
        openai = build_draft_provider(None, "sk-openai")
        self.assertIsInstance(openai, LLMDraftProvider)
        self.assertEqual(openai.vendor, "openai")
        self.assertIn("gpt", openai.name, "the provider name must identify the model")

    def test_anthropic_wins_when_both_are_present(self) -> None:
        """Same precedence as the classifier's build_provider. Two AI surfaces
        disagreeing about which vendor to use would be a nasty surprise."""
        both = build_draft_provider("sk-anthropic", "sk-openai")
        self.assertEqual(both.vendor, "anthropic")


class WireFormatTests(unittest.TestCase):
    """A malformed request is rejected by the vendor, not by us, so the shape
    has to be right the first time -- there is no test that catches it later."""

    def _capture(self, response: dict) -> tuple[OutreachDraft, dict]:
        sent: dict = {}

        def fake_post(url: str, payload: dict, headers: dict) -> dict:
            sent["url"], sent["payload"], sent["headers"] = url, payload, headers
            return response

        with mock.patch("app.ai_copilot._post_json", fake_post):
            provider = self._provider()
            return provider.draft(CONTEXT), sent

    def _provider(self) -> LLMDraftProvider:  # pragma: no cover - overridden
        raise NotImplementedError


class AnthropicWireTests(WireFormatTests):
    def _provider(self) -> LLMDraftProvider:
        return LLMDraftProvider("sk-a", vendor="anthropic", model="claude-sonnet-4-5")

    def _response(self) -> dict:
        return {
            "content": [
                {
                    "type": "tool_use",
                    "name": "record_draft",
                    "input": {
                        "language": "en",
                        "tone": "polite",
                        "message_body": "Your renewal did not go through. Please complete it below.",
                    },
                }
            ]
        }

    def test_the_tool_call_is_forced(self) -> None:
        """Without tool_choice the model may answer in prose, and prose has no
        schema to constrain it."""
        _, sent = self._capture(self._response())
        self.assertEqual(
            sent["payload"]["tool_choice"], {"type": "tool", "name": "record_draft"}
        )

    def test_the_schema_offers_nowhere_to_put_a_url(self) -> None:
        """The strongest guarantee in this file. A model cannot supply a link
        because the schema has no field for one -- that is structural, unlike
        a prompt telling it not to."""
        _, sent = self._capture(self._response())
        schema = sent["payload"]["tools"][0]["input_schema"]
        self.assertEqual(set(schema["properties"]), {"language", "tone", "message_body"})
        self.assertFalse(schema["additionalProperties"])

    def test_it_authenticates_and_versions(self) -> None:
        _, sent = self._capture(self._response())
        self.assertEqual(sent["headers"]["x-api-key"], "sk-a")
        self.assertIn("anthropic-version", sent["headers"])

    def test_it_reads_the_tool_use_block(self) -> None:
        draft, _ = self._capture(self._response())
        self.assertEqual(draft.tone, "polite")
        self.assertIn("renewal", draft.message_body)

    def test_a_response_with_no_tool_block_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._capture({"content": [{"type": "text", "text": "sure, here you go"}]})


class OpenAIWireTests(WireFormatTests):
    def _provider(self) -> LLMDraftProvider:
        return LLMDraftProvider("sk-o", vendor="openai", model="gpt-4.1-mini")

    def _response(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "language": "hinglish",
                                "tone": "reassuring",
                                "message_body": "Aapka renewal complete nahi hua. Neeche se pay karein.",
                            }
                        )
                    }
                }
            ]
        }

    def test_strict_schema_lists_every_property_as_required(self) -> None:
        """OpenAI's strict mode rejects a schema with an optional declared
        property outright. The classifier lost a whole run to this once."""
        _, sent = self._capture(self._response())
        schema = sent["payload"]["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(set(schema["schema"]["required"]), set(schema["schema"]["properties"]))

    def test_it_parses_the_json_content(self) -> None:
        draft, _ = self._capture(self._response())
        self.assertEqual(draft.language, "hinglish")


class GuardrailsHoldAgainstAModelTests(unittest.TestCase):
    """The point of the whole design: none of the safety is in the provider."""

    def _copilot_returning(self, body: str) -> OutreachCopilot:
        class Jailbroken:
            name = "jailbroken"

            def draft(self, context: OutreachContext) -> OutreachDraft:
                return OutreachDraft(language="en", tone="polite", message_body=body)

        return OutreachCopilot(Jailbroken())

    def test_a_model_supplied_url_is_blocked(self) -> None:
        preview = self._copilot_returning(
            "Please settle your dues right away at www.not-razorpay.example today"
        ).generate_preview(case(), LINK)
        self.assertFalse(preview.approved)
        self.assertIn("model_supplied_url", preview.blocked_reasons)

    def test_an_invented_discount_is_blocked(self) -> None:
        preview = self._copilot_returning(
            "Good news, we can give you 20% off your renewal if you pay today."
        ).generate_preview(case(), LINK)
        self.assertFalse(preview.approved)
        self.assertIn("unapproved_offer_or_discount", preview.blocked_reasons)

    def test_coercion_is_blocked(self) -> None:
        preview = self._copilot_returning(
            "Final warning: pay immediately or your account will be suspended shortly."
        ).generate_preview(case(), LINK)
        self.assertFalse(preview.approved)
        self.assertIn("coercive_dunning_language", preview.blocked_reasons)

    def test_the_link_is_appended_by_code_not_by_the_model(self) -> None:
        preview = self._copilot_returning(
            "Your subscription renewal did not complete. Please finish the payment."
        ).generate_preview(case(), LINK)
        self.assertTrue(preview.approved)
        self.assertNotIn(LINK, preview.message_body)
        self.assertIn(LINK, preview.final_message)

    def test_a_preview_is_never_a_delivery(self) -> None:
        preview = self._copilot_returning(
            "Your subscription renewal did not complete. Please finish the payment."
        ).generate_preview(case(), LINK)
        self.assertTrue(preview.preview_only)


class DegradationTests(unittest.TestCase):
    def test_a_provider_that_raises_falls_back_to_the_template(self) -> None:
        """A vendor timing out must not fail the merchant's request. They asked
        to see a draft, not to hear about our vendor."""

        class Broken:
            name = "broken"

            def draft(self, context: OutreachContext) -> OutreachDraft:
                raise TimeoutError("vendor did not answer")

        preview = OutreachCopilot(Broken()).generate_preview(case(), LINK)
        self.assertTrue(preview.approved)
        self.assertEqual(preview.drafted_by, "template_after_provider_error")

    def test_a_structurally_invalid_draft_is_refused_not_patched(self) -> None:
        class Garbage:
            name = "garbage"

            def draft(self, context: OutreachContext) -> dict:
                return {"language": "klingon", "tone": "menacing", "message_body": "x"}

        preview = OutreachCopilot(Garbage()).generate_preview(case(), LINK)
        self.assertFalse(preview.approved)
        self.assertIn("invalid_structured_draft", preview.blocked_reasons)


class AttributionTests(unittest.TestCase):
    """A lookup must never be presentable as a model."""

    def test_the_template_says_it_is_a_template(self) -> None:
        preview = OutreachCopilot().generate_preview(case(), LINK)
        self.assertEqual(preview.drafted_by, "template")

    def test_a_model_draft_names_the_model(self) -> None:
        class Fake:
            name = "llm:openai:gpt-4.1-mini"

            def draft(self, context: OutreachContext) -> OutreachDraft:
                return OutreachDraft(
                    language="en",
                    tone="polite",
                    message_body="Your renewal did not complete. Please finish the payment.",
                )

        preview = OutreachCopilot(Fake()).generate_preview(case(), LINK)
        self.assertEqual(preview.drafted_by, "llm:openai:gpt-4.1-mini")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
