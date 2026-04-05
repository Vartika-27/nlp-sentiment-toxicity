"""
Sentiment Analysis Pipeline — VADER
=====================================
Predict sentiment labels and evaluate model performance using VADER
(Valence Aware Dictionary and sEntiment Reasoner) with sklearn metrics.
"""

import csv
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSITIVE = "Positive"
NEGATIVE = "Negative"
NEUTRAL  = "Neutral"
LABELS   = (POSITIVE, NEGATIVE, NEUTRAL)

# VADER compound thresholds (Hutto & Gilbert 2014)
POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05

DEFAULT_OUTPUT_PATH = Path("sentiment_results.csv")

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_ANALYZER = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class SentimentResult:
    """Prediction output for a single text."""
    text:     str
    label:    str
    score:    float   # VADER compound score in [-1, 1]


@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics across a labelled dataset."""
    accuracy:          float
    precision_macro:   float
    recall_macro:      float
    f1_macro:          float
    classification_report: str   # full per-class breakdown


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_sentiment(text: Optional[str]) -> SentimentResult:
    """
    Predict the sentiment of *text* using VADER.

    Parameters
    ----------
    text:
        Input string. ``None`` or blank strings are handled gracefully
        and classified as Neutral with a compound score of 0.0.

    Returns
    -------
    SentimentResult
        Dataclass with ``label`` (Positive / Negative / Neutral)
        and ``score`` (VADER compound value in [-1, 1]).
    """
    if not isinstance(text, str) or not text.strip():
        logger.debug("Empty or non-string input — defaulting to Neutral.")
        return SentimentResult(text=text or "", label=NEUTRAL, score=0.0)

    scores  = _ANALYZER.polarity_scores(text)
    compound = round(scores["compound"], 4)

    if compound >= POSITIVE_THRESHOLD:
        label = POSITIVE
    elif compound <= NEGATIVE_THRESHOLD:
        label = NEGATIVE
    else:
        label = NEUTRAL

    return SentimentResult(text=text, label=label, score=compound)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    texts:  list[str],
    labels: list[str],
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    save_csv: bool = True,
) -> EvaluationMetrics:
    """
    Run VADER over *texts*, compare predictions against *labels*, and
    compute classification metrics.

    Parameters
    ----------
    texts:
        Raw input strings to classify.
    labels:
        Ground-truth sentiment labels (``"Positive"``, ``"Negative"``,
        or ``"Neutral"``).
    output_path:
        File path for the CSV results file.
    save_csv:
        Set to ``False`` to skip writing results to disk (useful in tests).

    Returns
    -------
    EvaluationMetrics
        Dataclass containing accuracy, macro-averaged precision / recall /
        F1, and the full ``sklearn`` classification report string.

    Raises
    ------
    ValueError
        If *texts* and *labels* differ in length or are empty.
    """
    if not texts or not labels:
        raise ValueError("texts and labels must be non-empty lists.")
    if len(texts) != len(labels):
        raise ValueError(
            f"texts ({len(texts)}) and labels ({len(labels)}) must have the same length."
        )

    invalid = set(labels) - set(LABELS)
    if invalid:
        raise ValueError(
            f"Unknown label(s) found: {invalid}. Allowed values: {LABELS}"
        )

    logger.info("Running VADER predictions on %d samples …", len(texts))
    results: list[SentimentResult] = [predict_sentiment(t) for t in texts]
    predictions = [r.label for r in results]

    # --- sklearn metrics ---
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    report = classification_report(
        labels, predictions, target_names=list(LABELS), zero_division=0
    )

    metrics = EvaluationMetrics(
        accuracy=round(accuracy, 4),
        precision_macro=round(precision, 4),
        recall_macro=round(recall, 4),
        f1_macro=round(f1, 4),
        classification_report=report,
    )

    _log_metrics(metrics)

    if save_csv:
        _save_results_csv(results, labels, Path(output_path))

    return metrics


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _log_metrics(metrics: EvaluationMetrics) -> None:
    """Emit a structured summary of *metrics* to the logger."""
    logger.info("─" * 50)
    logger.info("Accuracy  : %.4f", metrics.accuracy)
    logger.info("Precision : %.4f (macro)", metrics.precision_macro)
    logger.info("Recall    : %.4f (macro)", metrics.recall_macro)
    logger.info("F1-Score  : %.4f (macro)", metrics.f1_macro)
    logger.info("─" * 50)
    logger.info("Per-class report:\n%s", metrics.classification_report)


def _save_results_csv(
    results:     list[SentimentResult],
    true_labels: list[str],
    path:        Path,
) -> None:
    """
    Persist per-sample predictions alongside ground-truth labels to *path*.

    CSV columns: text, true_label, predicted_label, compound_score, correct
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "true_label", "predicted_label", "compound_score", "correct"])
        for result, true in zip(results, true_labels):
            writer.writerow([
                result.text,
                true,
                result.label,
                result.score,
                result.label == true,
            ])

    logger.info("Results saved → %s", path.resolve())


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_texts = [
        "I absolutely love this product! It exceeded all my expectations.",
        "This is the worst experience I have ever had. Totally disappointed.",
        "The item arrived on time.",
        "Fantastic quality and super fast shipping. Highly recommend!",
        "Not great, not terrible. It's just okay.",
        "I hate waiting so long for a mediocre result.",
        "The support team was incredibly helpful and friendly.",
        "Meh. Nothing special about it.",
        "Brilliant! Would buy again without hesitation.",
        "Absolute garbage. Complete waste of money.",
    ]

    sample_labels = [
        POSITIVE,
        NEGATIVE,
        NEUTRAL,
        POSITIVE,
        NEUTRAL,
        NEGATIVE,
        POSITIVE,
        NEUTRAL,
        POSITIVE,
        NEGATIVE,
    ]

    # --- single prediction demo ---
    print("=" * 60)
    print("  Single-text demo — predict_sentiment()")
    print("=" * 60)
    for text in sample_texts[:3]:
        result = predict_sentiment(text)
        print(f"  Text  : {result.text[:60]!r}")
        print(f"  Label : {result.label}  |  Score : {result.score}")
        print()

    # --- batch evaluation ---
    print("=" * 60)
    print("  Batch evaluation — evaluate_model()")
    print("=" * 60)
    metrics = evaluate_model(
        texts=sample_texts,
        labels=sample_labels,
        output_path="sentiment_results.csv",
    )

    print(f"\n  Accuracy  : {metrics.accuracy}")
    print(f"  Precision : {metrics.precision_macro}  (macro)")
    print(f"  Recall    : {metrics.recall_macro}  (macro)")
    print(f"  F1-Score  : {metrics.f1_macro}  (macro)")
    print(f"\n{metrics.classification_report}")