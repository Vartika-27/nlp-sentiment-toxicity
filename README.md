# 🧠 Multi-Layer NLP System

### Sentiment Analysis + Toxic Intent Detection

---

## 🚀 Overview

This project implements a **multi-layer Natural Language Processing (NLP) system** that analyzes user-generated text from two critical dimensions:

* **Emotional Tone (Sentiment)**
* **Behavioral Intent (Toxicity)**

Unlike traditional NLP systems, this project introduces a **Combined Insight Layer** that interprets deeper meaning such as **sarcasm, manipulation, and harmful intent**.

---

## 🎯 Problem Statement

Most NLP models answer:

> *“What is the sentiment?”*

This system goes further:

> *“What does this text actually mean in context?”*

---

## 🏗️ System Architecture

```
User Input
    ↓
Text Preprocessing
    ↓
┌───────────────────────┐
│  Sentiment Layer      │ → Positive / Negative / Neutral
└───────────────────────┘
    ↓
┌───────────────────────┐
│  Toxicity Layer       │ → Toxic / Non-Toxic
└───────────────────────┘
    ↓
┌────────────────────────────┐
│  Combined Insight Layer    │ → Final Interpretation
└────────────────────────────┘
```

---

## 🔍 Key Features

✨ Multi-layer NLP pipeline
✨ **Custom Toggle:** Dynamically switch between **Classic Baseline Models** and **State-of-the-Art Hugging Face Transformers**. 
✨ Generates **interpretable insights** based on emotional combinations.
✨ Modular and scalable codebase design.
✨ Highly-aesthetic, interactive **Streamlit Dashboard**.

---

## 🧠 Model Details

### 🔹 Sentiment Analysis
Users can toggle between two models in the UI:
1. **Classic Baseline:** VADER (Valence Aware Dictionary and sEntiment Reasoner) - Rule-based approach.
2. **Advanced Transformer:** `cardiffnlp/twitter-roberta-base-sentiment` - Deep learning approach for nuanced emotional detection.

### 🔹 Toxicity Detection
Users can toggle between two models:
1. **Classic Baseline:** TF-IDF + Logistic Regression - Supervised ML trained on custom seed text.
2. **Advanced Transformer:** `martin-ha/toxic-comment-model` (Toxic-BERT) - State-of-the-Art binary sequence classification.

---

### 🔥 Combined Insight Layer (Core Innovation)

| Sentiment | Toxicity  | Interpretation                     |
| --------- | --------- | ---------------------------------- |
| Positive  | Non-Toxic | Healthy expression                 |
| Negative  | Non-Toxic | Constructive criticism             |
| Positive  | Toxic     | Sarcastic / Manipulative tone      |
| Negative  | Toxic     | Direct harmful or abusive content  |
| Neutral   | Toxic     | Subtle or disguised harmful intent |

---

## 📁 Project Structure

```text
nlp-sentiment-toxicity/
│
├── data/
├── src/
│   ├── data/
│   ├── models/
│   ├── layers/
│   └── utils/
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_sentiment.py
│   ├── test_toxicity.py
│   └── test_combined.py
│
├── app/
│   └── app.py
├── reports/
├── results/
│
├── README.md
├── requirement.txt
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-link>
cd nlp-sentiment-toxicity
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirement.txt
```

---

## 🧪 Run Tests

Our pipeline is constantly tested for regressions.

```bash
python tests/test_pipeline.py
python tests/test_sentiment.py
python tests/test_toxicity.py
python tests/test_combined.py
```

---

## 🖥️ Run Demo App

Spin up the interactive Streamlit dashboard:

```bash
streamlit run app/app.py
```

---

## 📊 Sample Output

**Input:**

```text
Wow great job... idiot
```

**Output:**

```text
Sentiment: Positive  
Toxicity: Toxic  
Insight: Sarcastic / Manipulative positivity
```

---

## 📈 Future Scope

* Train baseline models on large-scale datasets (Jigsaw, SST-2) instead of dummy data.
* Improve sarcasm detection edge cases.
* Add multilingual support.
* Deploy as REST API for independent microservices.

---

## ⚠️ Limitations

* VADER struggles with nuanced sarcasm compared to the Transformer option.
* Running Hugging Face Transformers might be slightly slower strictly on CPU environments.
* Baseline Logistic Regression depends heavily on TF-IDF feature quality.

---

## 🎯 Key Takeaways

✔ Multi-layer analysis instead of single prediction.
✔ Combines emotion + intent dynamically.
✔ Produces human-interpretable insights.
✔ Now boasts an extensible architecture accommodating both Baseline and SOTA Transformer models.

---

## 👩‍💻 Author

Developed as part of a **B.Tech Project-Based Learning (PBL)** initiative.
Focused on **NLP + Cybersecurity Applications**.

---

⭐ *If you found this project interesting, consider starring the repository!*
