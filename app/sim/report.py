"""One command that prints every number this project claims.

    python -m app.sim.report

With ANTHROPIC_API_KEY or OPENAI_API_KEY set, the classifier rows use a real
model. Without one it falls back to the keyword provider and says so, because
a demo that silently swaps in a rule table and calls it AI is the thing this
whole harness exists to make impossible.

Every policy figure below is a mean across twelve seeds with a 95% confidence
interval, and every comparison is paired -- see ``app/sim/multiseed`` for why.
An earlier version of this report quoted single-seed point estimates to the
rupee. Several of the gaps it printed were narrower than the seed-to-seed
standard deviation, which is a polite way of saying it reported noise.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

from app.classifier import (
    KeywordProvider,
    LLMProvider,
    RootCauseClassifier,
    build_provider,
)
from app.sim.classifier_eval import evaluate, evaluate_published, evaluate_unseen
from app.sim.multiseed import (
    SEEDS,
    Series,
    paired,
    render_intervals,
    render_paired,
    sweep,
)
from app.sim.policies import (
    DoNothingPolicy,
    RetryThriceBackoffPolicy,
    RetryThriceImmediatePolicy,
    StaticDunningPolicy,
)
from app.sim.railpulse_policy import RailPulsePolicy
from app.sim.razorpay_corpus import CORPUS, DOC_SOURCES, EXCLUDED, RETRIEVED
from app.sim.world import WorldConfig

CASES = 500

BASELINES = (
    DoNothingPolicy,
    RetryThriceImmediatePolicy,
    RetryThriceBackoffPolicy,
    StaticDunningPolicy,
)


def _rule(title: str) -> str:
    return f"\n{title}\n{'=' * len(title)}"


def _step(message: str) -> None:
    """Progress on stderr, flushed.

    A live provider makes real HTTP calls, so a full report can run for
    minutes. Without this it prints a section header and then sits silent,
    which is indistinguishable from being hung.
    """
    print(f"  · {message}", file=sys.stderr, flush=True)


def _world_section(
    label: str,
    config: WorldConfig,
    make_provider: Callable[[], LLMProvider],
    shared_cache: dict,
    live: bool,
) -> None:
    print(_rule(label))
    factories = [
        *BASELINES,
        lambda: RailPulsePolicy(label="railpulse (no classifier)"),
        lambda: RailPulsePolicy(
            label="railpulse + classifier",
            normaliser=RootCauseClassifier(provider=make_provider(), cache=shared_cache),
        ),
    ]
    _step(
        f"{label.split()[0].lower()}: {len(factories)} policies x {len(SEEDS)} seeds"
        + ("  — calling the model" if live else "")
    )
    series: list[Series] = sweep(config, factories, seeds=SEEDS, cases=CASES)
    print(render_intervals(series))
    print()
    print(render_paired([paired(series[-1], other) for other in series[:-1]]))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    anthropic = os.getenv("ANTHROPIC_API_KEY")
    openai = os.getenv("OPENAI_API_KEY")

    def make_provider() -> LLMProvider:
        return build_provider(anthropic, openai)

    provider = make_provider()
    live = not isinstance(provider, KeywordProvider)
    # One cache for the whole report. Across twelve seeds and two worlds the
    # simulator emits only 76 distinct (code, message) pairs, so sharing turns
    # ~12,000 classifications into 76 HTTP calls. Counters stay per-classifier.
    shared_cache: dict = {}

    print(_rule("RailPulse — measured recovery"))
    print(
        f"classifier provider : {provider.name}"
        f"{'' if live else '  (NO API KEY SET — this is the rule-based floor, not a model)'}"
    )
    print(f"cases per world     : {CASES}   seeds: {len(SEEDS)}  ({SEEDS[0]}-{SEEDS[-1]})")
    print(
        "\nBoth policies and baselines face the identical world for a given seed.\n"
        "The world's recovery function is latent: no policy can read when a\n"
        "customer becomes solvent or when an outage clears, only act and observe.\n"
        "Figures are means over seeds; ± and ranges are 95% confidence intervals."
    )

    _world_section("DEFAULT WORLD", WorldConfig(), make_provider, shared_cache, live)
    _world_section(
        "SECOND WORLD (shifted parameters: a robustness check, not an independent test set)",
        WorldConfig().shifted(),
        make_provider,
        shared_cache,
        live,
    )

    print(_rule("CLASSIFIER — in-world messages"))
    _step("scoring classifier on in-world messages")
    print(evaluate(RootCauseClassifier(provider=make_provider(), cache=shared_cache)).as_text())

    print(_rule("CLASSIFIER — phrasings absent from the simulator"))
    print(
        "The number that matters. The keyword floor is a lookup written against\n"
        "the strings the simulator emits, so its in-world score is close to\n"
        "meaningless. Generalising to text nobody wrote a rule for is the job.\n"
    )
    _step("scoring classifier on unseen phrasings")
    # Deliberately NOT sharing the cache here: these phrasings appear nowhere
    # else, so there is nothing to reuse, and a fresh classifier keeps the
    # counters for this section clean.
    print(evaluate_unseen(RootCauseClassifier(provider=make_provider())).as_text())

    print(_rule("CLASSIFIER — Razorpay's own published error vocabulary"))
    print(
        "Text neither author of this project wrote. Every figure above is\n"
        f"measured against prose I wrote; these {len(CORPUS)} (code, description) pairs are\n"
        "copied verbatim from Razorpay's public error documentation, retrieved\n"
        f"{RETRIEVED}:\n"
        + "".join(f"  {url}\n" for url in DOC_SOURCES)
        + f"\n{len(EXCLUDED)} further published codes are excluded as genuinely ambiguous to\n"
        "label, listed with reasons in app/sim/razorpay_corpus.py. I assigned the\n"
        "cause labels; Razorpay publishes only the code and the description, so\n"
        "this is real provider text with my judgement on top -- not labelled\n"
        "production data, and the difference is stated in that module."
    )
    _step("scoring classifier on Razorpay's published codes (as a webhook arrives)")
    print("\n-- with the failure code present, as a real webhook carries it --")
    print(evaluate_published(RootCauseClassifier(provider=make_provider())).as_text())
    _step("scoring classifier on Razorpay's published codes (prose only)")
    print(
        "\n-- code withheld, prose only: the case the model exists for --\n"
        "   A lookup has nothing to match on here, so whatever survives is the\n"
        "   model reading language rather than a rule firing."
    )
    print(
        evaluate_published(
            RootCauseClassifier(provider=make_provider()), withhold_code=True
        ).as_text()
    )

    print(_rule("HONESTY NOTE"))
    print(
        "These are synthetic worlds, not merchant data. The claim is a\n"
        "controlled comparison between policies under identical conditions --\n"
        "not a forecast of real recovery for any merchant. Retry pricing is a\n"
        "flat per-attempt fee and does not model card-scheme excessive-retry\n"
        "penalties, which would penalise the retry-heavy baselines further.\n"
        "\n"
        "The opt-out column is not priced into net recovered. Static dunning's\n"
        "rupee figure therefore flatters it: it books the revenue and none of\n"
        "the cost of the customers it burned to earn it."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
