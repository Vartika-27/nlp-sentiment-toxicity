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
✨ Combines **rule-based + ML models**
✨ Generates **interpretable insights**
✨ Modular and scalable design
✨ Demo-ready system (Streamlit)

---

## 🧠 Model Details

### 🔹 Sentiment Analysis

* Model: **VADER**
* Type: Rule-based
* Output:

  * Positive
  * Negative
  * Neutral

---

### 🔹 Toxicity Detection

* Model: **TF-IDF + Logistic Regression**
* Type: Supervised ML
* Output:

  * Toxic
  * Non-Toxic

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

```
nlp-sentiment-toxicity/
│
├── data/
├── src/
│   ├── data/
│   ├── models/
│   ├── layers/
│   └── utils/
│
├── app/
├── reports/
├── results/
│
├── test_pipeline.py
├── test_sentiment.py
├── test_toxicity.py
├── test_combined.py
│
├── README.md
├── requirements.txt
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
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Run Components

```bash
python test_pipeline.py
python test_sentiment.py
python test_toxicity.py
python test_combined.py
```

---

## 🖥️ Run Demo App

```bash
streamlit run app/app.py
```

---

## 📊 Sample Output

**Input:**

```
Wow great job... idiot
```

**Output:**

```
Sentiment: Positive  
Toxicity: Toxic  
Insight: Sarcastic / Manipulative positivity
```

---

## 📈 Future Scope

* Fine-tune **BERT / DistilBERT**
* Train on large-scale datasets (Jigsaw, SST-2)
* Improve sarcasm detection
* Add multilingual support
* Deploy as REST API

---

## ⚠️ Limitations

* VADER struggles with nuanced sarcasm
* Logistic Regression depends on feature quality
* Small dataset → lower confidence scores

---

## 🎯 Key Takeaways

✔ Multi-layer analysis instead of single prediction
✔ Combines emotion + intent
✔ Produces human-interpretable insights
✔ Strong foundation for advanced NLP systems

---

## 👩‍💻 Author

Developed as part of a **B.Tech Project-Based Learning (PBL)** initiative
Focused on **NLP + Cybersecurity Applications**

---

⭐ *If you found this project interesting, consider starring the repository!*
