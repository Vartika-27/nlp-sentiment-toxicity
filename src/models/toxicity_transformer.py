"""
Toxicity Detection Model — Transformer Model
==========================================
Predict toxicity using a Hugging Face Transformer model.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

TOXIC     = "Toxic"
NON_TOXIC = "Non-Toxic"

@dataclass
class PredictionResult:
    text:        str
    label:       str    # "Toxic" | "Non-Toxic"
    probability: float  # P(Toxic) in [0, 1]

_MODEL_NAME = "martin-ha/toxic-comment-model"
_pipeline = None

def train_model(texts: list[str], labels: list[str]):
    # Maintain API compatibility with baseline, but transformers are pre-trained
    logger.info("train_model called, but Transformer model is already pre-trained. Skipping.")
    pass

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info(f"Loading transformer model: {_MODEL_NAME}")
        # device=-1 for CPU by default
        _pipeline = pipeline("text-classification", model=_MODEL_NAME, tokenizer=_MODEL_NAME)
    return _pipeline

def predict_toxicity(text: Optional[str]) -> PredictionResult:
    if not isinstance(text, str) or not text.strip():
        return PredictionResult(text=text or "", label=NON_TOXIC, probability=0.0)

    pipe = _get_pipeline()
    result = pipe(text, truncation=True, max_length=512)[0]
    
    # martin-ha/toxic-comment-model output labels are typically "toxic" and "non-toxic"
    hf_label = result["label"].lower()
    score = result["score"]
    
    if hf_label == "toxic":
        prob_toxic = score
        label = TOXIC
    else:
        prob_toxic = 1.0 - score
        label = NON_TOXIC

    return PredictionResult(text=text, label=label, probability=round(prob_toxic, 4))
