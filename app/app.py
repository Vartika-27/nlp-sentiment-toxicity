"""
NLP Analysis Dashboard — Streamlit App
=======================================
Combines text preprocessing, sentiment analysis (VADER), toxicity detection
(TF-IDF + Logistic Regression), and insight generation into a single
interactive demo interface.
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="NLP Insight Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Imports ─────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

# Allow running from the project root where the sibling modules live
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pipeline import clean_text
from src.models.sentiment_baseline import predict_sentiment as predict_sentiment_base, POSITIVE, NEGATIVE, NEUTRAL
from src.models.toxicity_baseline import predict_toxicity as predict_toxicity_base, train_model, TOXIC, NON_TOXIC
from src.layers.combined_insight import generate_insight

from src.models.sentiment_transformer import predict_sentiment as predict_sentiment_hf
from src.models.toxicity_transformer import predict_toxicity as predict_toxicity_hf

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: #0b0f1a;
        color: #e8e6f0;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero banner ── */
    .hero {
        text-align: center;
        padding: 3rem 1rem 2rem;
    }
    .hero-eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.2em;
        color: #7c6af7;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: clamp(2rem, 5vw, 3.4rem);
        font-weight: 800;
        line-height: 1.1;
        background: linear-gradient(135deg, #e8e6f0 30%, #7c6af7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.75rem;
    }
    .hero-sub {
        color: #8a8799;
        font-size: 1rem;
        font-weight: 400;
        max-width: 540px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Input card ── */
    .input-card {
        background: #131929;
        border: 1px solid #1f2a40;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin: 1.5rem 0;
    }
    .card-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        color: #5a5670;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* ── Streamlit textarea override ── */
    .stTextArea textarea {
        background: #0d1120 !important;
        border: 1px solid #252f45 !important;
        border-radius: 10px !important;
        color: #e8e6f0 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.9rem !important;
        resize: none !important;
        caret-color: #7c6af7;
    }
    .stTextArea textarea:focus {
        border-color: #7c6af7 !important;
        box-shadow: 0 0 0 3px rgba(124,106,247,0.15) !important;
    }

    /* ── Analyze button ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7c6af7, #5b48e8) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.04em;
        padding: 0.7rem 1.5rem !important;
        transition: opacity 0.2s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
    }

    /* ── Divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #1a2236;
        margin: 2rem 0;
    }

    /* ── Result grid cards ── */
    .result-card {
        background: #131929;
        border: 1px solid #1f2a40;
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        height: 100%;
    }
    .result-card-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.18em;
        color: #5a5670;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .result-value {
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .result-score {
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #5a5670;
        margin-top: 0.35rem;
    }

    /* ── Sentiment colours ── */
    .val-positive { color: #34d399; }
    .val-negative { color: #f87171; }
    .val-neutral  { color: #93c5fd; }

    /* ── Toxicity colours ── */
    .val-toxic    { color: #fb923c; }
    .val-nontoxic { color: #34d399; }

    /* ── Insight block ── */
    .insight-block {
        background: linear-gradient(135deg, #16203a 0%, #1a1535 100%);
        border: 1px solid #3a2f7a;
        border-left: 4px solid #7c6af7;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin: 1.5rem 0;
    }
    .insight-tag {
        display: inline-block;
        background: #7c6af7;
        color: #fff;
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        border-radius: 4px;
        padding: 0.2rem 0.55rem;
        margin-bottom: 0.75rem;
    }
    .insight-text {
        font-size: 1rem;
        font-weight: 600;
        line-height: 1.7;
        color: #cdc9e8;
    }

    /* ── Cleaned text expandable ── */
    .clean-text-box {
        background: #0d1120;
        border: 1px dashed #252f45;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        color: #6b6885;
        line-height: 1.6;
        word-break: break-word;
    }

    /* ── Progress bars ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7c6af7, #a78bfa) !important;
        border-radius: 99px !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.75rem !important;
        color: #5a5670 !important;
        letter-spacing: 0.1em;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2.5rem 0 1rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        color: #2e3550;
        letter-spacing: 0.1em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Seed the toxicity model (cached so it only trains once) ─────────────────
@st.cache_resource(show_spinner=False)
def _load_toxicity_model():
    seed_texts = [
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
    seed_labels = [TOXIC] * 10 + [NON_TOXIC] * 10
    train_model(seed_texts, seed_labels)

_load_toxicity_model()

# ── Helper Function: Render Analysis Results ──────────────────────────────────
def render_analysis_results(sent_result, tox_result, insight_result, cleaned, model_choice_labels):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p class="card-label" style="margin-bottom:1rem;">Analysis results</p>',
        unsafe_allow_html=True,
    )

    col_sent, col_tox = st.columns(2, gap="medium")

    sent_css = {
        POSITIVE: "val-positive",
        NEGATIVE: "val-negative",
        NEUTRAL:  "val-neutral",
    }.get(sent_result.label, "val-neutral")

    sent_bar_pct = int((sent_result.score + 1) / 2 * 100)
    sent_sign    = "+" if sent_result.score >= 0 else ""

    with col_sent:
        st.markdown(
            f'''
            <div class="result-card">
                <div class="result-card-label">😶 Sentiment</div>
                <div class="result-value {sent_css}">{sent_result.label}</div>
                <div class="result-score">compound {sent_sign}{sent_result.score:.4f}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.progress(sent_bar_pct)

    tox_css = "val-toxic" if tox_result.label == TOXIC else "val-nontoxic"
    tox_bar_pct = int(tox_result.probability * 100)

    with col_tox:
        st.markdown(
            f'''
            <div class="result-card">
                <div class="result-card-label">☣️ Toxicity</div>
                <div class="result-value {tox_css}">{tox_result.label}</div>
                <div class="result-score">P(toxic) {tox_result.probability:.4f}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.progress(tox_bar_pct)

    st.markdown(
        f'''
        <div class="insight-block">
            <span class="insight-tag">{insight_result.tag}</span>
            <div class="insight-text">💡 {insight_result.insight}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    with st.expander("VIEW PREPROCESSED TEXT", expanded=False):
        display_cleaned = cleaned if cleaned else "⚠️ Nothing remained after preprocessing."
        st.markdown(
            f'<div class="clean-text-box">{display_cleaned}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("VIEW RAW SCORES", expanded=False):
        import pandas as pd
        score_df = pd.DataFrame(
            {
                "Module": model_choice_labels,
                "Predicted Label": [sent_result.label, tox_result.label],
                "Score": [
                    f"{sent_result.score:+.4f} (compound)",
                    f"{tox_result.probability:.4f} P(toxic)",
                ],
            }
        )
        st.dataframe(score_df, use_container_width=True, hide_index=True)


# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">🔬 NLP Analysis Suite</div>
        <h1 class="hero-title">Insight Engine</h1>
        <p class="hero-sub">
            Paste any text below. The pipeline preprocesses it, scores
            sentiment &amp; toxicity, and surfaces a behavioural insight
            in under a second.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_compare = st.tabs(["Single Analysis", "Model Comparison"])

with tab_single:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Model Engine</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Choose Prediction Backend:",
        ["Baseline (VADER + LogReg)", "Advanced (Hugging Face Transformers)"],
        horizontal=True,
        label_visibility="collapsed",
        key="single_radio"
    )

    st.markdown('<div class="card-label" style="margin-top: 1rem;">Input text</div>', unsafe_allow_html=True)

    user_text_single = st.text_area(
        label="input_text_single",
        label_visibility="collapsed",
        placeholder='e.g. "Your product is absolutely fantastic - just kidding, it\'s terrible."',
        height=130,
        key="user_input_single",
    )

    analyze_clicked_single = st.button("⚡ Analyse Text", use_container_width=True, key="btn_single")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_clicked_single:
        raw = user_text_single.strip()
        if not raw:
            st.warning("Please enter some text before analysing.")
        else:
            with st.spinner("Running pipeline…"):
                cleaned = clean_text(raw)
                if "Advanced" in model_choice:
                    sent_result = predict_sentiment_hf(raw)
                    tox_result = predict_toxicity_hf(cleaned)
                    module_labels = ["RoBERTa HF Sentiment", "Transformer Toxicity"]
                else:
                    sent_result = predict_sentiment_base(raw)
                    tox_result = predict_toxicity_base(cleaned)
                    module_labels = ["VADER Sentiment", "LogReg Classifier"]
                insight_result = generate_insight(sent_result.label, tox_result.label)
            
            render_analysis_results(sent_result, tox_result, insight_result, cleaned, module_labels)


with tab_compare:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label" style="margin-top: 0.5rem;">Input text for comparison</div>', unsafe_allow_html=True)

    user_text_comp = st.text_area(
        label="input_text_comp",
        label_visibility="collapsed",
        placeholder='e.g. "Your product is absolutely fantastic - just kidding, it\'s terrible."',
        height=130,
        key="user_input_comp",
    )

    compare_clicked = st.button("⚡ Compare Models", use_container_width=True, key="btn_compare")
    st.markdown("</div>", unsafe_allow_html=True)

    if compare_clicked:
        raw = user_text_comp.strip()
        if not raw:
            st.warning("Please enter some text before analysing.")
        else:
            with st.spinner("Running both pipelines simultaneously…"):
                cleaned = clean_text(raw)
                
                # Run Baseline
                base_sent = predict_sentiment_base(raw)
                base_tox = predict_toxicity_base(cleaned)
                base_insight = generate_insight(base_sent.label, base_tox.label)
                
                # Run Transformers
                hf_sent = predict_sentiment_hf(raw)
                hf_tox = predict_toxicity_hf(cleaned)
                hf_insight = generate_insight(hf_sent.label, hf_tox.label)

            col_left, col_right = st.columns(2, gap="large")
            
            with col_left:
                st.markdown("<h3 style='text-align: center; color: #cdc9e8; font-family: Syne, sans-serif;'>Baseline Pipeline</h3>", unsafe_allow_html=True)
                render_analysis_results(base_sent, base_tox, base_insight, cleaned, ["VADER Sentiment", "LogReg Classifier"])
                
            with col_right:
                st.markdown("<h3 style='text-align: center; color: #cdc9e8; font-family: Syne, sans-serif;'>Transformer Pipeline</h3>", unsafe_allow_html=True)
                render_analysis_results(hf_sent, hf_tox, hf_insight, cleaned, ["RoBERTa HF Sentiment", "Transformer Toxicity"])

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-footer" style="margin-top: 3rem;">NLP INSIGHT ENGINE &nbsp;·&nbsp; CUSTOMIZABLE ML & HF PIPELINE</div>',
    unsafe_allow_html=True,
)