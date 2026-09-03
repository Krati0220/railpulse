"""The evaluation set that isn't mine.

Every other accuracy number in this project is measured against text I wrote,
which is a ceiling on what any of them can prove. This corpus is copied from
Razorpay's public error documentation, so the strings themselves come from
outside the project entirely.

That only helps if the corpus stays honest, and there are two specific ways it
could quietly stop being. It could drift into overlapping the simulator's own
message table, at which point it is measuring memorisation again. Or its
labels could bend toward whatever the classifier happens to answer, at which
point it is measuring nothing. The tests below are aimed at both, and at the
one mechanical property the whole "prose only" claim rests on: that withholding
the code actually withholds it.
"""

from __future__ import annotations

import unittest

from app.classifier import CanonicalCode, KeywordProvider, RootCauseClassifier
from app.models import PaymentRail
from app.sim.classifier_eval import ACCEPTABLE, DANGEROUS, evaluate_published
from app.sim.razorpay_corpus import CORPUS, DOC_SOURCES, EXCLUDED, by_rail
from app.sim.world import ISSUER_MESSAGES, Cause


class CorpusIntegrityTests(unittest.TestCase):
    def test_it_is_not_empty_and_covers_most_causes(self) -> None:
        self.assertGreaterEqual(len(CORPUS), 20)
        covered = {entry.cause for entry in CORPUS}
        # Every cause the engine acts on differently should be represented, or
        # the score is an average over a shape that does not match the problem.
        self.assertGreaterEqual(len(covered), 6, f"only {len(covered)} causes represented")

    def test_every_label_is_gradeable(self) -> None:
        """A cause with no entry in ACCEPTABLE would silently score 0% forever."""
        for entry in CORPUS:
            with self.subTest(code=entry.code):
                self.assertIn(entry.cause, ACCEPTABLE)
                self.assertTrue(ACCEPTABLE[entry.cause])

    def test_no_duplicate_code_and_rail(self) -> None:
        """The same code on two rails is legitimate -- Razorpay documents
        bank_technical_error on both, with different text. The same code twice
        on one rail is a copy-paste error that would double-weight it."""
        seen = [(entry.code, entry.rail) for entry in CORPUS]
        self.assertEqual(len(seen), len(set(seen)))

    def test_descriptions_are_real_sentences_not_placeholders(self) -> None:
        for entry in CORPUS:
            with self.subTest(code=entry.code):
                self.assertGreater(len(entry.description), 40)
                self.assertTrue(entry.description.endswith("."))
                self.assertNotIn("TODO", entry.description)

    def test_both_rails_are_present(self) -> None:
        self.assertTrue(by_rail(PaymentRail.CARD))
        self.assertTrue(by_rail(PaymentRail.UPI_AUTOPAY))

    def test_the_sources_are_recorded(self) -> None:
        """A corpus that cannot be traced back to its source is just more prose
        I wrote, which is the exact thing it exists to not be."""
        self.assertTrue(DOC_SOURCES)
        for url in DOC_SOURCES:
            self.assertTrue(url.startswith("https://razorpay.com/docs/"))

    def test_exclusions_are_reasoned_and_not_in_the_corpus(self) -> None:
        """Excluding an ambiguous case is legitimate; excluding one the
        classifier gets wrong is not. The reasons are the audit trail."""
        codes = {entry.code for entry in CORPUS}
        for excluded, reason in EXCLUDED.items():
            with self.subTest(code=excluded):
                self.assertGreater(len(reason), 40, "an exclusion needs a stated reason")
                base = excluded.split(" (")[0]
                self.assertNotIn(base, codes - {"gateway_technical_error", "payment_timed_out"})


class NotTheSimulatorsTextTests(unittest.TestCase):
    """The property that makes this corpus worth anything.

    If these strings overlapped the simulator's own message table, the score
    would be measuring whether a rule was written for a string somebody had
    already read -- which is precisely the criticism the corpus exists to
    answer.
    """

    def test_no_description_appears_in_the_simulator(self) -> None:
        simulator_text = {
            message.lower() for messages in ISSUER_MESSAGES.values() for message in messages
        }
        for entry in CORPUS:
            with self.subTest(code=entry.code):
                self.assertNotIn(entry.description.lower(), simulator_text)

    def test_no_simulator_phrase_is_embedded_in_a_description(self) -> None:
        """Substring containment, not just equality: a published description
        that happened to contain a simulator phrase verbatim would let the
        keyword lookup match it for the wrong reason."""
        for messages in ISSUER_MESSAGES.values():
            for message in messages:
                for entry in CORPUS:
                    with self.subTest(phrase=message, code=entry.code):
                        self.assertNotIn(message.lower(), entry.description.lower())


class WithholdingTests(unittest.TestCase):
    """The 'prose only' number depends on the code really being gone."""

    def test_withholding_actually_withholds_the_code(self) -> None:
        seen: list[tuple[str | None, str]] = []

        class Recording:
            """Records what it was asked, then declines to answer.

            Raising a caught exception rather than returning a sentinel keeps
            this independent of Classification's shape -- the classifier
            handles it as a provider error and moves on, which is all this
            test needs.
            """

            name = "recording"

            def classify(self, failure_code, issuer_message, **kwargs):
                seen.append((failure_code, issuer_message))
                raise ValueError("declined on purpose")

        classifier = RootCauseClassifier(provider=Recording())
        evaluate_published(classifier, withhold_code=True)
        self.assertTrue(seen)
        self.assertTrue(
            all(code is None for code, _ in seen),
            "a failure code reached the classifier despite withhold_code=True",
        )

    def test_the_code_is_passed_when_not_withheld(self) -> None:
        classifier = RootCauseClassifier(provider=KeywordProvider())
        with_code = evaluate_published(classifier, withhold_code=False)
        self.assertGreater(with_code.answered, 0)


class KeywordFloorTests(unittest.TestCase):
    """What a rule table is worth on somebody else's vocabulary."""

    def _floor(self, withhold: bool):
        return evaluate_published(
            RootCauseClassifier(provider=KeywordProvider()), withhold_code=withhold
        )

    def test_the_rule_table_does_not_generalise(self) -> None:
        """The premise of the whole section. If a lookup scored well on real
        provider text, the model would have no job and the README should say
        so instead."""
        self.assertLess(
            self._floor(withhold=False).accuracy_overall,
            0.6,
            "the keyword floor now generalises to real provider text; the "
            "argument for a model needs re-examining, not the threshold",
        )

    def test_stripping_the_code_leaves_a_lookup_with_almost_nothing(self) -> None:
        self.assertLess(self._floor(withhold=True).accuracy_overall, 0.2)

    def test_the_floor_refuses_rather_than_guesses(self) -> None:
        """Low coverage is the correct failure mode for a rule table meeting
        unfamiliar text. Confident wrong answers would be the bad outcome, and
        a dangerous confusion is the worst kind."""
        self.assertEqual(self._floor(withhold=False).dangerous, 0)
        self.assertEqual(self._floor(withhold=True).dangerous, 0)


class DangerousConfusionTests(unittest.TestCase):
    def test_the_dangerous_pairs_are_gradeable_against_this_corpus(self) -> None:
        """A dangerous-confusion count of zero means nothing if none of the
        corpus's causes could produce one."""
        corpus_causes = {entry.cause for entry in CORPUS}
        reachable = {cause for cause, _ in DANGEROUS if cause in corpus_causes}
        self.assertTrue(
            reachable,
            "no corpus entry can produce a dangerous confusion, so the column is decorative",
        )

    def test_a_dead_instrument_read_as_low_balance_is_dangerous(self) -> None:
        """The specific harm: a retry storm against a card that will never
        clear. Named here so the definition cannot quietly loosen."""
        self.assertIn(
            (Cause.INSTRUMENT_DEAD, CanonicalCode.INSUFFICIENT_FUNDS), DANGEROUS
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
