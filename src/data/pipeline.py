"""
NLP Text Preprocessing Pipeline
================================
A clean, modular, and production-ready preprocessing pipeline for NLP text data.
"""
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import re
import logging
from typing import Optional

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK resource bootstrap (idempotent – safe to call multiple times)
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = [
    ("tokenizers/punkt",          "punkt"),
    ("tokenizers/punkt_tab",      "punkt_tab"),
    ("corpora/stopwords",         "stopwords"),
    ("corpora/wordnet",           "wordnet"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]


def _download_nltk_resources() -> None:
    """Download required NLTK resources if they are not already present."""
    for resource_path, resource_name in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource_name)
            nltk.download(resource_name, quiet=True)


_download_nltk_resources()

# ---------------------------------------------------------------------------
# Module-level singletons (initialised once for performance)
# ---------------------------------------------------------------------------
_LEMMATIZER   = WordNetLemmatizer()
_STOP_WORDS   = set(stopwords.words("english"))

# Pre-compiled regular expressions
_RE_URL       = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_HTML      = re.compile(r"<[^>]+>")
_RE_PUNCT     = re.compile(r"[^\w\s]")
_RE_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Individual, single-responsibility cleaning steps
# ---------------------------------------------------------------------------

def to_lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Strip HTTP/HTTPS and bare www URLs from *text*."""
    return _RE_URL.sub(" ", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML / XML tags from *text*."""
    return _RE_HTML.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from *text*."""
    return _RE_PUNCT.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters into a single space."""
    return _RE_WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize *text* into a list of word tokens using NLTK's word tokenizer."""
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove English stop-words from *tokens*."""
    return [t for t in tokens if t not in _STOP_WORDS]


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to a WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Lemmatize *tokens* using WordNet, leveraging POS tags for accuracy.

    Parameters
    ----------
    tokens:
        List of word strings (already lowercased and stripped of punctuation).

    Returns
    -------
    list[str]
        Lemmatized token list.
    """
    pos_tags = pos_tag(tokens)
    return [
        _LEMMATIZER.lemmatize(token, _wordnet_pos(tag))
        for token, tag in pos_tags
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: Optional[str]) -> str:
    """
    Apply the full preprocessing pipeline to a single text string.

    Steps applied (in order):
        1. Guard against ``None`` / non-string input.
        2. Lowercase.
        3. Remove URLs.
        4. Remove HTML tags.
        5. Remove punctuation.
        6. Normalize whitespace.
        7. Tokenize (NLTK ``word_tokenize``).
        8. Remove stop-words.
        9. Lemmatize with WordNet (POS-aware).

    Parameters
    ----------
    text:
        Raw input string (may be ``None`` or empty).

    Returns
    -------
    str
        Cleaned, space-joined string of lemmatized tokens.
        Returns an empty string for ``None`` / blank inputs.
    """
    # --- edge-case guard ---
    if not isinstance(text, str):
        logger.debug("Received non-string input (%s); returning empty string.", type(text))
        return ""

    text = text.strip()
    if not text:
        return ""

    # --- sequential cleaning ---
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)

    # --- token-level operations ---
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    column_name: str,
    output_column: Optional[str] = None,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply :func:`clean_text` to every row of *column_name* in *df*.

    Parameters
    ----------
    df:
        Input ``pandas.DataFrame``.
    column_name:
        Name of the column that contains raw text.
    output_column:
        Name of the column where cleaned text will be stored.
        Defaults to ``"{column_name}_cleaned"``.
    inplace:
        If ``True``, modify *df* in place; otherwise work on a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new cleaned-text column appended.

    Raises
    ------
    ValueError
        If *column_name* is not present in *df*.
    TypeError
        If *df* is not a ``pandas.DataFrame``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__!r}.")

    if column_name not in df.columns:
        raise ValueError(
            f"Column {column_name!r} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if output_column is None:
        output_column = f"{column_name}_cleaned"

    result = df if inplace else df.copy()

    logger.info(
        "Preprocessing column %r → %r  (%d rows).",
        column_name, output_column, len(result),
    )

    result[output_column] = result[column_name].apply(clean_text)

    logger.info("Preprocessing complete.")
    return result


# ---------------------------------------------------------------------------
# Quick smoke-test (only executed when run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "Check out https://example.com for <b>amazing</b> deals!!!",
        "The quick brown foxes are jumping over the lazy dogs.",
        "<p>NLP pipelines   strip   extra   whitespace & HTML tags.</p>",
        None,
        "",
        "   ",
        "Running, ran, runs — all forms lemmatized correctly.",
    ]

    print("=" * 60)
    print("  Single-string demo — clean_text()")
    print("=" * 60)
    for raw in sample_texts:
        cleaned = clean_text(raw)
        print(f"  IN : {raw!r}")
        print(f"  OUT: {cleaned!r}")
        print()

    print("=" * 60)
    print("  DataFrame demo — preprocess_dataframe()")
    print("=" * 60)
    demo_df = pd.DataFrame({"review": sample_texts})
    result_df = preprocess_dataframe(demo_df, column_name="review")
    print(result_df.to_string(index=False))