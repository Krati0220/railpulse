#!/usr/bin/env python3
"""Exchange real packets with Razorpay's test-mode API.

    RAZORPAY_KEY_ID=rzp_test_... RAZORPAY_KEY_SECRET=... \\
        python scripts/verify_razorpay_live.py

The gateway adapter is at 99% coverage against a fake transport. That proves
the code does what the tests say when the tests hold the other end of the
socket, and nothing at all about whether Razorpay agrees. This closes that gap
the only way it can be closed: by talking to Razorpay.

It uses **test mode only** and refuses to run against a live key, because
every check below creates a real payment link and then revokes it. Against
live credentials that would be a payable URL on a real merchant account.

Nothing here runs in CI. It needs credentials, it makes network calls, and a
test that does either is not a test -- it is a check you run deliberately,
which is what this is.

Exit status is 0 only if every check passes, so it is usable as a gate.
"""

from __future__ import annotations

import os
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.gateway import (
    PermanentGatewayError,
    RazorpayPaymentLinkGateway,
)

PASS = "  ok   "
FAIL = "  FAIL "


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        print(f"{PASS if condition else FAIL} {name}{(' — ' + detail) if detail else ''}")
        if not condition:
            self.failures.append(name)
        return condition


def main() -> int:
    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")

    if not key_id or not key_secret:
        print(
            "Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET first.\n"
            "Test keys are free: Razorpay Dashboard -> Account & Settings -> API Keys,\n"
            "with the mode switch set to Test. A test key id starts with 'rzp_test_'."
        )
        return 2

    if not key_id.startswith("rzp_test_"):
        print(
            f"Refusing to run: {key_id[:12]}... is not a test key.\n"
            "This script creates and cancels real payment links. Against live\n"
            "credentials that is a payable URL on a real merchant account."
        )
        return 2

    checks = Checks()
    gateway = RazorpayPaymentLinkGateway(key_id, key_secret)
    reference = f"railpulse-verify-{uuid.uuid4().hex[:12]}"

    print(f"\nRazorpay test mode · {key_id[:16]}…\nreference: {reference}\n")

    # ---------------------------------------------------------------- create
    try:
        link = gateway.create_payment_link(
            amount_paise=49900,
            reference_id=reference,
            description="RailPulse adapter verification (test mode)",
        )
    except Exception as exc:
        checks.check("create_payment_link", False, f"{type(exc).__name__}: {exc}")
        print("\nCannot continue without a link.")
        return 1

    checks.check("create_payment_link returns an id", bool(link.id), link.id)
    checks.check(
        "the short_url is a real Razorpay link",
        link.short_url.startswith("https://") and "rzp" in link.short_url,
        link.short_url,
    )
    checks.check("the link is created, not already paid", link.status == "created", link.status)

    # ----------------------------------------------------------- idempotency
    # The claim this makes: the adapter derives a stable key from the
    # reference id, so a retry of the same logical action reuses it. If that
    # is wrong, Razorpay creates a SECOND live link for one invoice and only
    # the second id comes back -- the first is orphaned, uncancellable, and
    # payable after the payment has already recovered. Asserted against the
    # provider rather than against a fake, because the provider is what
    # enforces it.
    try:
        repeat = gateway.create_payment_link(
            amount_paise=49900,
            reference_id=reference,
            description="RailPulse adapter verification (test mode)",
        )
        checks.check(
            "a repeated create returns the SAME link, not a second one",
            repeat.id == link.id,
            f"{link.id} vs {repeat.id}",
        )
    except Exception as exc:
        checks.check("repeated create is idempotent", False, f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- cancel
    try:
        status = gateway.cancel_payment_link(link.id)
        checks.check("cancel_payment_link revokes it", status == "cancelled", status)
    except Exception as exc:
        checks.check("cancel_payment_link", False, f"{type(exc).__name__}: {exc}")
        print(f"\n!! {link.short_url} is still live. Cancel it from the dashboard.")
        return 1

    # A cancel that is not idempotent at the provider would break the retry
    # sweep in RecoveryService.retry_failed_link_cancels, which re-issues
    # cancels whose response was lost.
    time.sleep(1)
    try:
        again = gateway.cancel_payment_link(link.id)
        checks.check("cancelling twice is harmless", again == "cancelled", again)
    except PermanentGatewayError as exc:
        # Razorpay may refuse a second cancel outright. That is also safe --
        # the link is revoked either way -- but the sweep must treat it as
        # settled rather than as a failure to retry, so it is worth knowing.
        checks.check(
            "cancelling twice is harmless",
            True,
            f"provider refuses a second cancel ({exc}); link is revoked, sweep must not retry",
        )

    # ----------------------------------------------------- error handling
    # A 4xx must arrive as PermanentGatewayError, not as a transient one: the
    # retry policy branches on exactly this, and misclassifying a permanent
    # rejection means retrying it three times for nothing.
    try:
        gateway.cancel_payment_link("plink_definitely_not_real")
        checks.check("an unknown link id is a permanent error", False, "no exception raised")
    except PermanentGatewayError as exc:
        checks.check("an unknown link id is a permanent error", True, str(exc)[:80])
    except Exception as exc:
        checks.check(
            "an unknown link id is a permanent error",
            False,
            f"got {type(exc).__name__} instead: {exc}",
        )

    print()
    if checks.failures:
        print(f"{len(checks.failures)} check(s) failed: {', '.join(checks.failures)}")
        return 1
    print("All checks passed against Razorpay test mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
