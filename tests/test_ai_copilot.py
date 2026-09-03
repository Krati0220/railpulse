from __future__ import annotations

import unittest
from datetime import UTC, datetime

from app.ai_copilot import NonCompliantDraftProvider, OutreachCopilot
from app.models import FailureClass, PaymentRail, RecoveryCase, RecoveryCaseState


class UnsafeDraftProvider:
    def draft(self, _context):
        return {
            "language": "hinglish",
            "tone": "polite_urgent",
            "message_body": "Pay now at https://attacker.example/p and get 20% cashback, or legal action will begin.",
        }


class MalformedDraftProvider:
    def draft(self, _context):
        return {"language": "klingon", "tone": "shouty", "message_body": "hi"}


def recovery_case() -> RecoveryCase:
    return RecoveryCase(
        id="case_preview",
        logical_key="inv_preview",
        amount_paise=49900,
        failure_class=FailureClass.CUSTOMER_ACTION,
        rail=PaymentRail.UPI_AUTOPAY,
    )


class OutreachCopilotTests(unittest.TestCase):
    def test_safe_preview_has_one_policy_owned_payment_link(self) -> None:
        preview = OutreachCopilot().generate_preview(recovery_case(), "https://rzp.io/i/plink_demo_safe")
        self.assertTrue(preview.approved)
        self.assertEqual(preview.final_message.count("https://rzp.io/i/plink_demo_safe"), 1)
        self.assertIn("canonical_link_appended_by_code", preview.policy_checks)
        self.assertTrue(preview.preview_only)

    def test_unsafe_draft_is_blocked_before_the_link_is_appended(self) -> None:
        preview = OutreachCopilot(UnsafeDraftProvider()).generate_preview(
            recovery_case(), "https://rzp.io/i/plink_demo_safe"
        )
        self.assertFalse(preview.approved)
        self.assertIsNone(preview.final_message)
        self.assertIn("model_supplied_url", preview.blocked_reasons)
        self.assertIn("unapproved_offer_or_discount", preview.blocked_reasons)
        self.assertIn("coercive_dunning_language", preview.blocked_reasons)

    def test_malformed_draft_is_rejected_by_structured_validation(self) -> None:
        preview = OutreachCopilot(MalformedDraftProvider()).generate_preview(
            recovery_case(), "https://rzp.io/i/plink_demo_safe"
        )
        self.assertFalse(preview.approved)
        self.assertEqual(preview.blocked_reasons, ["invalid_structured_draft"])

    def test_a_non_canonical_link_is_never_appended(self) -> None:
        for url in (
            "http://rzp.io/i/plink",
            "https://rzp.io.evil.example/i/plink",
            "https://rzp.io",
            "",
        ):
            with self.subTest(url=url):
                preview = OutreachCopilot().generate_preview(recovery_case(), url)
                self.assertFalse(preview.approved)
                self.assertIn("invalid_canonical_payment_link", preview.blocked_reasons)
                self.assertIsNone(preview.final_message)

    def test_every_language_produces_an_approved_draft(self) -> None:
        for language in ("en", "hi", "hinglish"):
            with self.subTest(language=language):
                preview = OutreachCopilot().generate_preview(
                    recovery_case(), "https://rzp.io/i/plink_demo_safe", language=language
                )
                self.assertTrue(preview.approved)
                self.assertEqual(preview.language, language)


if __name__ == "__main__":
    unittest.main()


class GuardrailDemoTests(unittest.TestCase):
    """The validator has to be seen refusing something, not just described.

    Before this the only drafting provider emitted three hardcoded compliant
    strings, so OutreachPolicyValidator could never return a reason and the
    dashboard's blocked branch was unreachable code.
    """

    def _case(self) -> RecoveryCase:
        return RecoveryCase(
            id="case_guard",
            logical_key="inv_guard",
            state=RecoveryCaseState.LINK_SENT,
            amount_paise=49900,
            failure_class=FailureClass.CUSTOMER_ACTION,
            rail=PaymentRail.UPI_AUTOPAY,
            updated_at=datetime(2026, 8, 21, tzinfo=UTC),
        )

    def test_a_non_compliant_draft_trips_every_rule(self) -> None:
        preview = OutreachCopilot(NonCompliantDraftProvider()).generate_preview(
            self._case(), "https://rzp.io/i/plink_guard"
        )
        self.assertFalse(preview.approved)
        self.assertEqual(
            set(preview.blocked_reasons),
            {"model_supplied_url", "unapproved_offer_or_discount", "coercive_dunning_language"},
        )

    def test_a_blocked_draft_never_gets_a_sendable_message(self) -> None:
        """The refusal has to withhold the payload, not merely flag it."""
        preview = OutreachCopilot(NonCompliantDraftProvider()).generate_preview(
            self._case(), "https://rzp.io/i/plink_guard"
        )
        self.assertIsNone(preview.final_message)
        self.assertTrue(preview.preview_only)
        self.assertIsNotNone(preview.message_body, "the rejected text stays visible for audit")

    def test_the_compliant_provider_still_passes(self) -> None:
        preview = OutreachCopilot().generate_preview(
            self._case(), "https://rzp.io/i/plink_guard"
        )
        self.assertTrue(preview.approved)
        self.assertEqual(preview.blocked_reasons, [])
        self.assertIn("rzp.io", preview.final_message)
