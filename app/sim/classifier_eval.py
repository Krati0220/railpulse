"""Scores the classifier against the world's hidden ground truth.

The classifier only ever sees ``(failure_code, issuer_message)``. The world
knows the true :class:`Cause`. Comparing the two is the only honest way to say
what the model is worth, and it is deliberately reported separately from the
rupee scorecard: a classifier can be accurate and still not move money, and
conflating the two is how uplift claims get inflated.

Coverage matters as much as accuracy here. A classifier that answers only when
certain has high accuracy and leaves the engine blind; one that always answers
has full coverage and sends retries at dead cards. Both numbers are printed.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.classifier import CanonicalCode, RootCauseClassifier
from app.sim.razorpay_corpus import CORPUS
from app.sim.world import Cause, World, WorldConfig

#: Every canonical code that is a defensible answer for a given true cause.
#: Dead instruments legitimately surface as expired, closed or invalid-VPA, so
#: all three are correct; risk blocks may present as fraud or an open dispute.
ACCEPTABLE: dict[Cause, set[CanonicalCode]] = {
    Cause.INSUFFICIENT_FUNDS: {CanonicalCode.INSUFFICIENT_FUNDS},
    Cause.ISSUER_OUTAGE: {CanonicalCode.ISSUER_UNAVAILABLE, CanonicalCode.GATEWAY_ERROR},
    Cause.DO_NOT_HONOUR: {CanonicalCode.DO_NOT_HONOUR},
    Cause.TECHNICAL_DECLINE: {CanonicalCode.GATEWAY_ERROR, CanonicalCode.ISSUER_UNAVAILABLE},
    Cause.INSTRUMENT_DEAD: {
        CanonicalCode.CARD_EXPIRED,
        CanonicalCode.ACCOUNT_CLOSED,
        CanonicalCode.INVALID_VPA,
    },
    Cause.MANDATE_CANCELLED: {CanonicalCode.MANDATE_CANCELLED},
    Cause.RISK_BLOCKED: {CanonicalCode.SUSPECTED_FRAUD, CanonicalCode.CHARGEBACK_OPEN},
}

#: Confusions that cause real financial harm, as opposed to merely being wrong.
#: Calling a dead card "insufficient funds" produces a retry storm against an
#: instrument that will never clear; calling a fraud block anything else means
#: contacting a customer the system was supposed to leave alone.
DANGEROUS = {
    (Cause.INSTRUMENT_DEAD, CanonicalCode.INSUFFICIENT_FUNDS),
    (Cause.INSTRUMENT_DEAD, CanonicalCode.ISSUER_UNAVAILABLE),
    (Cause.INSTRUMENT_DEAD, CanonicalCode.GATEWAY_ERROR),
    (Cause.MANDATE_CANCELLED, CanonicalCode.INSUFFICIENT_FUNDS),
    (Cause.RISK_BLOCKED, CanonicalCode.INSUFFICIENT_FUNDS),
    (Cause.RISK_BLOCKED, CanonicalCode.GATEWAY_ERROR),
    (Cause.RISK_BLOCKED, CanonicalCode.ISSUER_UNAVAILABLE),
    (Cause.RISK_BLOCKED, CanonicalCode.DO_NOT_HONOUR),
}


#: Issuer strings that appear NOWHERE in the world's message table.
#:
#: This exists because the keyword provider is a hand-written lookup over the
#: exact phrases the simulator emits, so it scores ~93% on the in-world batch
#: and that number means almost nothing. Real acquirer feeds invent new
#: phrasings constantly -- new banks, new switches, Hinglish, typos, internal
#: error codes. Generalising to text nobody wrote a rule for is the entire
#: reason to put a model here, so it is measured separately and honestly.
UNSEEN_MESSAGES: tuple[tuple[str | None, str, Cause], ...] = (
    (None, "Balance kam hai, please recharge", Cause.INSUFFICIENT_FUNDS),
    (None, "AVAILABLE BAL LESS THAN TXN AMT", Cause.INSUFFICIENT_FUNDS),
    (None, "funds unavailable at this time", Cause.INSUFFICIENT_FUNDS),
    ("DECLINE-BAL", "customer a/c short", Cause.INSUFFICIENT_FUNDS),
    (None, "remitter bank offline", Cause.ISSUER_OUTAGE),
    (None, "RB not reachable, retry after some time", Cause.ISSUER_OUTAGE),
    (None, "switch timeout at beneficiary bank", Cause.ISSUER_OUTAGE),
    (None, "system under maintenance till 0400 hrs", Cause.ISSUER_OUTAGE),
    (None, "refer to card issuer", Cause.DO_NOT_HONOUR),
    (None, "txn refused - please contact your bank", Cause.DO_NOT_HONOUR),
    ("DECLINED_BY_ISSUER", "no reason provided by issuer", Cause.DO_NOT_HONOUR),
    (None, "internal error code 9999", Cause.TECHNICAL_DECLINE),
    (None, "unable to process at this moment", Cause.TECHNICAL_DECLINE),
    (None, "temporary glitch please try again", Cause.TECHNICAL_DECLINE),
    (None, "card blocked by issuer permanently", Cause.INSTRUMENT_DEAD),
    (None, "VPA does not exist", Cause.INSTRUMENT_DEAD),
    (None, "a/c dormant or closed", Cause.INSTRUMENT_DEAD),
    (None, "validity period over", Cause.INSTRUMENT_DEAD),
    (None, "UMN not found at NPCI", Cause.MANDATE_CANCELLED),
    (None, "standing instruction withdrawn by customer", Cause.MANDATE_CANCELLED),
    (None, "debit authorisation inactive", Cause.MANDATE_CANCELLED),
    (None, "txn flagged by AML rules", Cause.RISK_BLOCKED),
    (None, "velocity check breach - blocked", Cause.RISK_BLOCKED),
    (None, "issuer risk decline, do not reattempt", Cause.RISK_BLOCKED),
)


@dataclass(frozen=True)
class EvalReport:
    total: int
    answered: int
    correct: int
    dangerous: int
    per_cause: dict[str, tuple[int, int]]
    stats: dict[str, int | float]

    @property
    def coverage(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def accuracy_on_answered(self) -> float:
        return self.correct / self.answered if self.answered else 0.0

    @property
    def accuracy_overall(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def as_text(self) -> str:
        lines = [
            f"cases                : {self.total}",
            f"answered (coverage)  : {self.answered}  ({self.coverage:.1%})",
            f"correct / answered   : {self.correct}  ({self.accuracy_on_answered:.1%})",
            f"correct / all cases  : {self.accuracy_overall:.1%}",
            f"dangerous confusions : {self.dangerous}",
            "",
            "per true cause        correct/seen",
        ]
        for cause, (correct, seen) in sorted(self.per_cause.items()):
            rate = correct / seen if seen else 0.0
            lines.append(f"  {cause:<20} {correct:>4}/{seen:<5} {rate:>6.1%}")
        lines.append("")
        lines.append(f"provider stats       : {self.stats}")
        return "\n".join(lines)


def evaluate(
    classifier: RootCauseClassifier,
    config: WorldConfig | None = None,
    seed: int = 11,
    cases: int = 500,
) -> EvalReport:
    """Classify every payment in a world and grade against latent truth.

    Note the default seed differs from the scorecard's. The classifier is
    graded on a batch the policy comparison does not run on, so a classifier
    tuned against these messages cannot quietly inflate the rupee table.
    """
    world = World(config or WorldConfig(), seed=seed, cases=cases)
    answered = correct = dangerous = 0
    per_cause: dict[str, list[int]] = {}

    for payment_id in world.payment_ids:
        observation = world.observe(payment_id, world.failed_at(payment_id))
        truth = Cause(world.ledger(payment_id)["cause"])
        bucket = per_cause.setdefault(truth.value, [0, 0])
        bucket[1] += 1

        answer = classifier.normalise(observation.failure_code, observation.issuer_message)
        if answer is None:
            continue
        answered += 1
        code = CanonicalCode(answer)
        if code in ACCEPTABLE[truth]:
            correct += 1
            bucket[0] += 1
        elif (truth, code) in DANGEROUS:
            dangerous += 1

    return EvalReport(
        total=len(world.payment_ids),
        answered=answered,
        correct=correct,
        dangerous=dangerous,
        per_cause={k: (v[0], v[1]) for k, v in per_cause.items()},
        stats=classifier.stats(),
    )


def evaluate_unseen(classifier: RootCauseClassifier) -> EvalReport:
    """Grade on phrasings that appear nowhere in the simulator.

    This is the number worth quoting. The in-world score mostly measures
    whether someone wrote a rule for a string they had already read.
    """
    answered = correct = dangerous = 0
    per_cause: dict[str, list[int]] = {}

    for failure_code, message, truth in UNSEEN_MESSAGES:
        bucket = per_cause.setdefault(truth.value, [0, 0])
        bucket[1] += 1
        answer = classifier.normalise(failure_code, message)
        if answer is None:
            continue
        answered += 1
        code = CanonicalCode(answer)
        if code in ACCEPTABLE[truth]:
            correct += 1
            bucket[0] += 1
        elif (truth, code) in DANGEROUS:
            dangerous += 1

    return EvalReport(
        total=len(UNSEEN_MESSAGES),
        answered=answered,
        correct=correct,
        dangerous=dangerous,
        per_cause={k: (v[0], v[1]) for k, v in per_cause.items()},
        stats=classifier.stats(),
    )


def evaluate_published(
    classifier: RootCauseClassifier, *, withhold_code: bool = False
) -> EvalReport:
    """Grade on Razorpay's own published failure vocabulary.

    Every other figure in this module is measured against text I wrote, which
    caps what any of them can prove. These strings come from the payment
    provider's public error documentation, written by people who have never
    seen this codebase -- see ``app/sim/razorpay_corpus`` for exactly what that
    does and does not establish, including the fact that I assigned the labels.

    Two modes, because they answer different questions:

    ``withhold_code=False`` is what a real webhook looks like -- code and
    description both present. It grades the classifier as a whole, lookup
    included, and is the number a merchant would actually experience.

    ``withhold_code=True`` deletes the code and leaves only the prose. This is
    the case the model exists for: the lookup has nothing to match on, so
    whatever accuracy survives is the model reading language. A rule table
    scores near zero here by construction, which is the point.
    """
    answered = correct = dangerous = 0
    per_cause: dict[str, list[int]] = {}

    for entry in CORPUS:
        bucket = per_cause.setdefault(entry.cause.value, [0, 0])
        bucket[1] += 1
        answer = classifier.normalise(
            None if withhold_code else entry.code, entry.description
        )
        if answer is None:
            continue
        answered += 1
        code = CanonicalCode(answer)
        if code in ACCEPTABLE[entry.cause]:
            correct += 1
            bucket[0] += 1
        elif (entry.cause, code) in DANGEROUS:
            dangerous += 1

    return EvalReport(
        total=len(CORPUS),
        answered=answered,
        correct=correct,
        dangerous=dangerous,
        per_cause={k: (v[0], v[1]) for k, v in per_cause.items()},
        stats=classifier.stats(),
    )
