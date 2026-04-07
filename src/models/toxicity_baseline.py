"""
Toxicity Detection Model — TF-IDF + Logistic Regression
=========================================================
Train, predict, and evaluate a binary text toxicity classifier
using scikit-learn. Outputs per-sample probabilities, classification
metrics, and a confusion-matrix PNG.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

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
TOXIC     = "Toxic"
NON_TOXIC = "Non-Toxic"
LABELS    = (TOXIC, NON_TOXIC)

DEFAULT_CM_PATH = Path("confusion_matrix.png")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class PredictionResult:
    """Single-text prediction output."""
    text:        str
    label:       str    # "Toxic" | "Non-Toxic"
    probability: float  # P(Toxic) in [0, 1]


@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics for a labelled dataset."""
    accuracy:           float
    precision_macro:    float
    recall_macro:       float
    f1_macro:           float
    classification_report: str
    confusion_matrix:   np.ndarray


# ---------------------------------------------------------------------------
# Module-level model state
# ---------------------------------------------------------------------------
# The pipeline is intentionally module-level so predict_toxicity() can call
# it without requiring the caller to pass a model handle.
_pipeline: Optional[Pipeline] = None


def _require_trained_model() -> Pipeline:
    """Return the trained pipeline, raising clearly if it hasn't been fitted."""
    if _pipeline is None:
        raise RuntimeError(
            "Model is not trained yet. Call train_model(texts, labels) first."
        )
    return _pipeline


# ---------------------------------------------------------------------------
# 1. Training
# ---------------------------------------------------------------------------

def train_model(
    texts:  list[str],
    labels: list[str],
    *,
    max_features:   int   = 10_000,
    ngram_range:    tuple = (1, 2),
    C:              float = 1.0,
    max_iter:       int   = 1_000,
    random_state:   int   = 42,
) -> Pipeline:
    """
    Fit a TF-IDF → Logistic Regression pipeline on *texts* / *labels*.

    Parameters
    ----------
    texts:
        Raw training strings.
    labels:
        Binary ground-truth labels — each element must be ``"Toxic"`` or
        ``"Non-Toxic"``.
    max_features:
        Maximum vocabulary size passed to :class:`TfidfVectorizer`.
    ngram_range:
        N-gram range for the vectorizer (default unigrams + bigrams).
    C:
        Inverse regularisation strength for Logistic Regression.
    max_iter:
        Solver iteration cap.
    random_state:
        Reproducibility seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The fitted pipeline (also stored in the module-level ``_pipeline``).

    Raises
    ------
    ValueError
        If *texts* / *labels* are empty or mismatched in length, or if
        unknown label values are present.
    """
    global _pipeline

    _validate_inputs(texts, labels)

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents="unicode",
                    lowercase=True,
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ],
        verbose=False,
    )

    logger.info(
        "Training on %d samples  |  vocab=%d  |  ngrams=%s  |  C=%.3f",
        len(texts), max_features, ngram_range, C,
    )
    pipeline.fit(texts, labels)
    logger.info("Training complete.")

    _pipeline = pipeline
    return pipeline


# ---------------------------------------------------------------------------
# 2. Prediction
# ---------------------------------------------------------------------------

def predict_toxicity(text: Optional[str]) -> PredictionResult:
    """
    Predict whether *text* is toxic.

    Parameters
    ----------
    text:
        Raw input string. ``None`` / blank strings return ``"Non-Toxic"``
        with probability ``0.0`` and a debug log entry.

    Returns
    -------
    PredictionResult
        ``label`` is ``"Toxic"`` or ``"Non-Toxic"``; ``probability`` is
        P(Toxic) ∈ [0, 1].
    """
    model = _require_trained_model()

    if not isinstance(text, str) or not text.strip():
        logger.debug("Empty or non-string input — defaulting to Non-Toxic.")
        return PredictionResult(text=text or "", label=NON_TOXIC, probability=0.0)

    proba   = model.predict_proba([text])[0]                      # shape (n_classes,)
    classes = list(model.classes_)
    toxic_prob = float(proba[classes.index(TOXIC)])

    label = TOXIC if toxic_prob >= 0.5 else NON_TOXIC

    return PredictionResult(text=text, label=label, probability=round(toxic_prob, 4))


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    texts:  list[str],
    labels: list[str],
    *,
    cm_path:  Path | str = DEFAULT_CM_PATH,
    save_png: bool = True,
) -> EvaluationMetrics:
    """
    Evaluate the trained model on a labelled dataset.

    Parameters
    ----------
    texts:
        Input strings to classify.
    labels:
        Ground-truth labels (``"Toxic"`` / ``"Non-Toxic"``).
    cm_path:
        File path for the confusion-matrix PNG.
    save_png:
        Set to ``False`` to skip writing the PNG (useful in unit tests).

    Returns
    -------
    EvaluationMetrics
        Dataclass with accuracy, macro P/R/F1, full classification report,
        and the raw confusion matrix array.

    Raises
    ------
    RuntimeError
        If the model has not been trained yet.
    ValueError
        If *texts* / *labels* are mismatched or contain unknown labels.
    """
    _require_trained_model()
    _validate_inputs(texts, labels)

    logger.info("Evaluating on %d samples …", len(texts))
    predictions = [predict_toxicity(t).label for t in texts]

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    report = classification_report(
        labels, predictions, target_names=list(LABELS), zero_division=0
    )
    cm = confusion_matrix(labels, predictions, labels=list(LABELS))

    metrics = EvaluationMetrics(
        accuracy=round(float(accuracy), 4),
        precision_macro=round(float(precision), 4),
        recall_macro=round(float(recall), 4),
        f1_macro=round(float(f1), 4),
        classification_report=report,
        confusion_matrix=cm,
    )

    _log_metrics(metrics)

    if save_png:
        _save_confusion_matrix(cm, Path(cm_path))

    return metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(texts: list[str], labels: list[str]) -> None:
    """Raise ``ValueError`` for empty, mismatched, or invalid-label inputs."""
    if not texts or not labels:
        raise ValueError("texts and labels must be non-empty lists.")
    if len(texts) != len(labels):
        raise ValueError(
            f"texts ({len(texts)}) and labels ({len(labels)}) must have the same length."
        )
    invalid = set(labels) - set(LABELS)
    if invalid:
        raise ValueError(
            f"Unknown label(s): {invalid}. Allowed values: {set(LABELS)}"
        )


def _log_metrics(metrics: EvaluationMetrics) -> None:
    logger.info("─" * 50)
    logger.info("Accuracy  : %.4f", metrics.accuracy)
    logger.info("Precision : %.4f (macro)", metrics.precision_macro)
    logger.info("Recall    : %.4f (macro)", metrics.recall_macro)
    logger.info("F1-Score  : %.4f (macro)", metrics.f1_macro)
    logger.info("─" * 50)
    logger.info("Per-class report:\n%s", metrics.classification_report)


def _save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    """Render and save the confusion matrix as a PNG file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(LABELS))
    disp.plot(
        ax=ax,
        colorbar=True,
        cmap="Blues",
        values_format="d",
    )
    ax.set_title("Toxicity Detection — Confusion Matrix", fontsize=13, pad=14)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", path.resolve())


# ---------------------------------------------------------------------------
# Smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- toy dataset ---
    train_texts = [
        "You are an idiot and I hate you",
        "Go kill yourself, nobody likes you",
        "This is absolutely disgusting behaviour",
        "I will destroy you, you worthless piece of trash",
        "You're so stupid, stop talking",
        "Shut up you moron",
        "What an awful, hateful comment",
        "Get out of here you loser",
        "I hope you fail at everything in life",
        "You disgust me",
        "The weather is nice today",
        "Thanks for sharing that information",
        "I appreciate your help with this problem",
        "Great work on the project!",
        "Looking forward to seeing you at the event",
        "This is a wonderful community",
        "Have a great day everyone",
        "Your analysis was really insightful",
        "That recipe looks delicious!",
        "Congratulations on your achievement",
    ]

    train_labels = (
        [TOXIC] * 10
        + [NON_TOXIC] * 10
    )

    eval_texts = [
        "You are worthless and nobody cares about you",
        "What a lovely day for a walk in the park",
        "Stop being such an idiot all the time",
        "I really enjoyed reading your article",
        "You are completely useless",
        "Thanks for your kind words!",
    ]

    eval_labels = [TOXIC, NON_TOXIC, TOXIC, NON_TOXIC, TOXIC, NON_TOXIC]

    # 1. Train
    train_model(train_texts, train_labels)

    # 2. Single-text predictions
    print("=" * 60)
    print("  predict_toxicity() — individual samples")
    print("=" * 60)
    for sample in eval_texts:
        result = predict_toxicity(sample)
        print(f"  Text  : {result.text[:55]!r}")
        print(f"  Label : {result.label:<10}  P(Toxic) : {result.probability:.4f}")
        print()

    # 3. Batch evaluation
    print("=" * 60)
    print("  evaluate_model() — batch metrics")
    print("=" * 60)
    metrics = evaluate_model(eval_texts, eval_labels, cm_path="confusion_matrix.png")

    print(f"\n  Accuracy  : {metrics.accuracy}")
    print(f"  Precision : {metrics.precision_macro}  (macro)")
    print(f"  Recall    : {metrics.recall_macro}  (macro)")
    print(f"  F1-Score  : {metrics.f1_macro}  (macro)")
    print(f"\n{metrics.classification_report}")
    print(f"  Confusion matrix:\n{metrics.confusion_matrix}")