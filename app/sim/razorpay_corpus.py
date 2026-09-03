"""Razorpay's own published failure vocabulary, as an evaluation set.

Every other accuracy figure in this project is measured against text I wrote.
That is a hard ceiling on what any of them can prove: a classifier evaluated
on its author's prose is being asked whether it can read its author's mind.
The 24 "unseen phrasings" in ``classifier_eval`` narrow the problem -- no rule
was written for them -- but I still wrote them, so they carry my assumptions
about what an acquirer message looks like.

These do not. Every ``description`` below is copied verbatim from Razorpay's
public error documentation, which is to say: from the payment provider this
project integrates with, written by people who have never seen this codebase.

    https://razorpay.com/docs/errors/payments/cards/
    https://razorpay.com/docs/errors/payments/upi/

Retrieved 2026-09-03.

What this does and does not establish
-------------------------------------
It establishes that the classifier reads *real* provider text -- the exact
vocabulary a merchant integrating with Razorpay actually receives -- rather
than a stylised imitation of it.

It does **not** make this labelled production data, and the difference matters
enough to state plainly:

* **I assigned the ``cause`` labels.** Razorpay publishes the code and the
  description; the mapping onto this system's seven recovery causes is my
  judgement. A biased labelling would flatter the classifier, so the mappings
  are kept deliberately literal (a description that says "downtime" is an
  outage; one that says "expired" is a dead instrument) and every contestable
  case is excluded rather than argued for -- see ``EXCLUDED`` below, which is
  as much a part of this file's honesty as the entries that made it in.
* **It is a vocabulary, not a distribution.** Real traffic is dominated by a
  handful of these codes; this set weights each equally. So it measures whether
  the classifier can read the language, not what it would score on a merchant's
  actual mix.
* **These are documented strings, not raw acquirer output.** A real feed
  carries typos, bank-specific variants and switch-level codes that never reach
  a docs page.

The honest summary: this is the strongest evaluation available without a
merchant's data, and it is still not a merchant's data.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models import PaymentRail
from app.sim.world import Cause

DOC_SOURCES = (
    "https://razorpay.com/docs/errors/payments/cards/",
    "https://razorpay.com/docs/errors/payments/upi/",
)
RETRIEVED = "2026-09-03"


@dataclass(frozen=True)
class PublishedFailure:
    """One (code, description) pair exactly as Razorpay documents it."""

    code: str
    #: Verbatim. Not paraphrased, not shortened, not tidied.
    description: str
    rail: PaymentRail
    #: My label, not Razorpay's. See the module docstring.
    cause: Cause


CARD_FAILURES: tuple[PublishedFailure, ...] = (
    PublishedFailure(
        "gateway_technical_error",
        "There was a downtime on our partner bank due to which the payment has failed.",
        PaymentRail.CARD,
        Cause.ISSUER_OUTAGE,
    ),
    PublishedFailure(
        "bank_downtime",
        "There was a downtime on the customer's bank due to which the payment has failed.",
        PaymentRail.CARD,
        Cause.ISSUER_OUTAGE,
    ),
    PublishedFailure(
        "bank_technical_error",
        "There was a downtime on the customer's bank due to which the payment has failed.",
        PaymentRail.CARD,
        Cause.ISSUER_OUTAGE,
    ),
    PublishedFailure(
        "card_declined",
        "The payment was declined by the customer's bank, resulting in the transaction "
        "being unsuccessful.",
        PaymentRail.CARD,
        # A generic issuer refusal with no stated reason is exactly the
        # do-not-honour shape: sticky to the instrument, not to the customer.
        Cause.DO_NOT_HONOUR,
    ),
    PublishedFailure(
        "payment_failed",
        "The payment was declined by the customer's bank, resulting in unsuccessful "
        "transaction.",
        PaymentRail.CARD,
        Cause.DO_NOT_HONOUR,
    ),
    PublishedFailure(
        "insufficient_funds",
        "The payment did not go through because the customer's bank account did not have "
        "enough funds.",
        PaymentRail.CARD,
        Cause.INSUFFICIENT_FUNDS,
    ),
    PublishedFailure(
        "card_expired",
        "The payment could not be completed because the customer's card is expired.",
        PaymentRail.CARD,
        Cause.INSTRUMENT_DEAD,
    ),
    PublishedFailure(
        "card_not_enrolled",
        "The payment was unsuccessful as the card was not activated or enabled by the "
        "customer for online transactions.",
        PaymentRail.CARD,
        Cause.INSTRUMENT_DEAD,
    ),
    PublishedFailure(
        "card_disabled_for_online_payments",
        "The payment was unsuccessful as the card was not activated or enabled for online "
        "transactions.",
        PaymentRail.CARD,
        Cause.INSTRUMENT_DEAD,
    ),
    PublishedFailure(
        "debit_instrument_inactive",
        "The payment was unsuccessful as the card was not activated or enabled for online "
        "transactions.",
        PaymentRail.CARD,
        Cause.INSTRUMENT_DEAD,
    ),
    PublishedFailure(
        "debit_instrument_blocked",
        "The payment could not be processed due to the card being blocked by the customer "
        "or bank.",
        PaymentRail.CARD,
        Cause.INSTRUMENT_DEAD,
    ),
    PublishedFailure(
        "payment_risk_check_failed",
        "The transaction was unsuccessful as the customer's bank declined the payment, "
        "citing it as fraudulent.",
        PaymentRail.CARD,
        Cause.RISK_BLOCKED,
    ),
    PublishedFailure(
        "payment_timed_out",
        "The payment could not be completed as the customer exceeded the time limit for "
        "payment processing.",
        PaymentRail.CARD,
        Cause.TECHNICAL_DECLINE,
    ),
)

UPI_FAILURES: tuple[PublishedFailure, ...] = (
    PublishedFailure(
        "bank_technical_error",
        "The payment failed due to a downtime on the UPI provider.",
        PaymentRail.UPI_AUTOPAY,
        Cause.ISSUER_OUTAGE,
    ),
    PublishedFailure(
        "partner_bank_downtime",
        "The payment failed due to a downtime on our partner bank.",
        PaymentRail.UPI_AUTOPAY,
        Cause.ISSUER_OUTAGE,
    ),
    PublishedFailure(
        "partner_bank_technical_issues",
        "The payment could not be completed due to technical issues from the partner bank.",
        PaymentRail.UPI_AUTOPAY,
        Cause.ISSUER_OUTAGE,
    ),
    PublishedFailure(
        "insufficient_funds",
        "The payment did not go through because the customer's bank account did not have "
        "enough funds to complete the transaction.",
        PaymentRail.UPI_AUTOPAY,
        Cause.INSUFFICIENT_FUNDS,
    ),
    PublishedFailure(
        "invalid_vpa",
        "The payment was unsuccessful due to the customer not being a valid user on the "
        "UPI App.",
        PaymentRail.UPI_AUTOPAY,
        Cause.INSTRUMENT_DEAD,
    ),
    PublishedFailure(
        "vpa_resolution_failed",
        "The payment was unsuccessful due to a failure to process the transaction using "
        "the customer's UPI ID.",
        PaymentRail.UPI_AUTOPAY,
        Cause.INSTRUMENT_DEAD,
    ),
    PublishedFailure(
        "customer_bank_account_mismatch",
        "The payment was unsuccessful because the customer selected a different bank "
        "account that was not used during the time of registration.",
        PaymentRail.UPI_AUTOPAY,
        # An autopay mandate is bound to the account it was registered on. A
        # different account means the mandate no longer covers the payment.
        Cause.MANDATE_CANCELLED,
    ),
    PublishedFailure(
        "payment_declined",
        "The payment did not go through because the funds could not be debited from the "
        "customer's bank account.",
        PaymentRail.UPI_AUTOPAY,
        Cause.DO_NOT_HONOUR,
    ),
    PublishedFailure(
        "payment_collect_request_expired",
        "The payment could not be completed as the customer exceeded the time limit for "
        "payment processing.",
        PaymentRail.UPI_AUTOPAY,
        Cause.TECHNICAL_DECLINE,
    ),
)

CORPUS: tuple[PublishedFailure, ...] = CARD_FAILURES + UPI_FAILURES

#: Published codes deliberately left out, and why. Excluding a case because it
#: is genuinely ambiguous is legitimate; excluding it because the classifier
#: gets it wrong is not, so the reasons are recorded here and every one of them
#: is about the *label*, never about the answer.
EXCLUDED: dict[str, str] = {
    "payment_cancelled": (
        "The customer chose not to pay. There is no failure to recover from, so no "
        "recovery cause is the right answer."
    ),
    "incorrect_cvv": (
        "A data-entry mistake during checkout, not a state of the instrument or the "
        "customer's balance."
    ),
    "authentication_failed": (
        "'incorrect OTP or closed the browser' spans a retryable technical failure and a "
        "customer walking away. Genuinely two causes in one code."
    ),
    "transaction_limit_exceeded": (
        "A per-card ceiling behaves like insufficient funds for recovery purposes, but "
        "the money exists. Defensible either way, so not used to score anything."
    ),
    "invalid_otp": (
        "eMandate registration failure, upstream of the recovery lifecycle this system "
        "models."
    ),
    "credit_failed": "Razorpay lists the code without a description; there is no text to classify.",
    "gateway_technical_error (UPI)": (
        "Listed on the UPI page without a description. The card-rail entry of the same "
        "name has one and is included."
    ),
    "payment_timed_out (UPI)": (
        "Listed without a description distinct from the collect-request expiry, which is "
        "included."
    ),
}


def by_rail(rail: PaymentRail) -> tuple[PublishedFailure, ...]:
    return tuple(item for item in CORPUS if item.rail is rail)
