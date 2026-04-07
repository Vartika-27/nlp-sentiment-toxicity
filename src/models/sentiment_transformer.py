"""
Sentiment Analysis Pipeline — Transformer Model
=============================================
Predict sentiment labels using a Hugging Face Transformer model.
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

POSITIVE = "Positive"
NEGATIVE = "Negative"
NEUTRAL  = "Neutral"

@dataclass
class SentimentResult:
    text:     str
    label:    str
    score:    float   # Map pipeline score to a pseudocompound [-1, 1]

# Load cardiffnlp model which has 3 classes: LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info(f"Loading transformer model: {_MODEL_NAME}")
        # Using device=-1 for CPU by default, Streamlit will load this when accessed
        _pipeline = pipeline("sentiment-analysis", model=_MODEL_NAME, tokenizer=_MODEL_NAME)
    return _pipeline

def predict_sentiment(text: Optional[str]) -> SentimentResult:
    if not isinstance(text, str) or not text.strip():
        return SentimentResult(text=text or "", label=NEUTRAL, score=0.0)

    pipe = _get_pipeline()
    # Handle long texts by truncating silently to max_length
    result = pipe(text, truncation=True, max_length=512)[0]
    
    label_mapping = {
        "LABEL_0": NEGATIVE,
        "LABEL_1": NEUTRAL,
        "LABEL_2": POSITIVE
    }
    
    hf_label = result["label"]
    score = result["score"] # This is [0, 1] confidence
    
    final_label = label_mapping.get(hf_label, NEUTRAL)
    
    # Map back to a pseudo-vader compound for UI compatibility
    pseudo_compound = score if final_label == POSITIVE else (-score if final_label == NEGATIVE else 0.0)

    return SentimentResult(text=text, label=final_label, score=pseudo_compound)
