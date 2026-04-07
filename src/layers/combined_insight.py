"""
Insight Generator — Sentiment × Toxicity
=========================================
Combines a sentiment label (Positive / Negative / Neutral) with a toxicity
label (Toxic / Non-Toxic) to produce a human-readable behavioural insight.
"""

import logging
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid label sets (single source of truth — import from sibling modules
# if this lives inside a package)
# ---------------------------------------------------------------------------
SENTIMENT_LABELS = frozenset({"Positive", "Negative", "Neutral"})
TOXICITY_LABELS  = frozenset({"Toxic", "Non-Toxic"})

# ---------------------------------------------------------------------------
# Insight lookup table
# (sentiment, toxicity) → (short_tag, full_insight)
# ---------------------------------------------------------------------------
_INSIGHT_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("Positive", "Toxic"): (
        "Sarcastic / Manipulative",
        "Sarcastic or manipulative positivity — positive framing is likely "
        "weaponised to belittle, provoke, or deceive.",
    ),
    ("Positive", "Non-Toxic"): (
        "Healthy Positivity",
        "Healthy positive expression — genuine encouragement, praise, or "
        "constructive optimism.",
    ),
    ("Negative", "Toxic"): (
        "Harmful / Abusive",
        "Direct harmful or abusive content — negative sentiment combined with "
        "toxic language signals aggression or personal attacks.",
    ),
    ("Negative", "Non-Toxic"): (
        "Constructive Criticism",
        "Constructive criticism or complaint — negative tone without toxic "
        "intent, often actionable feedback or a legitimate grievance.",
    ),
    ("Neutral", "Toxic"): (
        "Covert Harm",
        "Potentially harmful neutral statement — neutral wording may mask "
        "passive aggression, dog-whistling, or subtle manipulation.",
    ),
    ("Neutral", "Non-Toxic"): (
        "Balanced / Informational",
        "Balanced or informational content — objective, matter-of-fact "
        "communication with no emotional charge or harmful intent.",
    ),
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InsightResult:
    """Full output produced by :func:`generate_insight`."""
    sentiment_label: str
    toxicity_label:  str
    tag:             str   # short category name
    insight:         str   # full human-readable explanation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_insight(
    sentiment_label: Optional[str],
    toxicity_label:  Optional[str],
) -> InsightResult:
    """
    Combine *sentiment_label* and *toxicity_label* into a behavioural insight.

    Parameters
    ----------
    sentiment_label:
        One of ``"Positive"``, ``"Negative"``, or ``"Neutral"``.
        Surrounding whitespace is stripped; matching is case-sensitive.
    toxicity_label:
        One of ``"Toxic"`` or ``"Non-Toxic"``.
        Surrounding whitespace is stripped; matching is case-sensitive.

    Returns
    -------
    InsightResult
        Frozen dataclass containing the normalised labels, a short ``tag``,
        and a full ``insight`` string.

    Raises
    ------
    ValueError
        If either label is ``None``, not a string, or not a recognised value.
    """
    sentiment_label, toxicity_label = _validate_and_normalise(
        sentiment_label, toxicity_label
    )

    tag, insight = _INSIGHT_MAP[(sentiment_label, toxicity_label)]

    result = InsightResult(
        sentiment_label=sentiment_label,
        toxicity_label=toxicity_label,
        tag=tag,
        insight=insight,
    )

    logger.debug(
        "Insight generated  |  sentiment=%r  toxicity=%r  tag=%r",
        sentiment_label, toxicity_label, tag,
    )

    return result


def describe_all_combinations() -> list[InsightResult]:
    """
    Return :class:`InsightResult` objects for every valid label combination.

    Useful for documentation, UI dropdowns, or test coverage.
    """
    return [
        generate_insight(sentiment, toxicity)
        for sentiment, toxicity in _INSIGHT_MAP
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_and_normalise(
    sentiment_label: Optional[str],
    toxicity_label:  Optional[str],
) -> tuple[str, str]:
    """Strip whitespace and validate both labels; raise ``ValueError`` if invalid."""
    if not isinstance(sentiment_label, str):
        raise ValueError(
            f"sentiment_label must be a string, got {type(sentiment_label).__name__!r}."
        )
    if not isinstance(toxicity_label, str):
        raise ValueError(
            f"toxicity_label must be a string, got {type(toxicity_label).__name__!r}."
        )

    sentiment_label = sentiment_label.strip()
    toxicity_label  = toxicity_label.strip()

    if sentiment_label not in SENTIMENT_LABELS:
        raise ValueError(
            f"Unknown sentiment_label {sentiment_label!r}. "
            f"Valid values: {sorted(SENTIMENT_LABELS)}"
        )
    if toxicity_label not in TOXICITY_LABELS:
        raise ValueError(
            f"Unknown toxicity_label {toxicity_label!r}. "
            f"Valid values: {sorted(TOXICITY_LABELS)}"
        )

    return sentiment_label, toxicity_label


# ---------------------------------------------------------------------------
# Smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    combinations = [
        ("Positive", "Toxic"),
        ("Positive", "Non-Toxic"),
        ("Negative", "Toxic"),
        ("Negative", "Non-Toxic"),
        ("Neutral",  "Toxic"),
        ("Neutral",  "Non-Toxic"),
    ]

    print("=" * 70)
    print("  generate_insight() — all label combinations")
    print("=" * 70)
    for sentiment, toxicity in combinations:
        result = generate_insight(sentiment, toxicity)
        print(f"\n  Sentiment : {result.sentiment_label}")
        print(f"  Toxicity  : {result.toxicity_label}")
        print(f"  Tag       : {result.tag}")
        print(f"  Insight   : {result.insight}")

    # --- edge-case handling ---
    print("\n" + "=" * 70)
    print("  Edge-case validation")
    print("=" * 70)
    bad_inputs = [
        (None, "Toxic"),
        ("Happy", "Toxic"),
        ("Positive", "Harmful"),
        ("", "Non-Toxic"),
    ]
    for s, t in bad_inputs:
        try:
            generate_insight(s, t)
        except ValueError as exc:
            print(f"  ValueError for ({s!r}, {t!r}): {exc}")