# 🕵️‍♂️ Fake Job Posting Detection using NLP

A real-world machine learning project that detects fraudulent job postings using natural language processing (NLP) and classification models.

---

## 📌 Problem Statement
Fake job postings are a growing threat to job seekers, often leading to scams or data theft. This project aims to build a machine learning model that can classify a job posting as real or fake based on its content.

---

## 🧠 Features
- End-to-end pipeline from raw data to model deployment
- Natural Language Processing using TF-IDF
- Logistic Regression classification model
- Model performance evaluated using precision, recall, F1-score
- Streamlit web app for live predictions

---

## 📁 Dataset
- **Source**: [Kaggle – Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size**: ~17,000 job listings
- **Target**: `fraudulent` (1 = Fake, 0 = Real)

---

## 📊 Tech Stack
- Python (Pandas, NumPy, Scikit-learn)
- NLP: TfidfVectorizer
- Model: Logistic Regression
- Deployment: Streamlit
- Model Storage: Joblib

---

## 🚀 Getting Started

### 🔧 Installation
```bash
git clone https://github.com/3-vaibhav/fake-job-detector.git
cd fake-job-detector
pip install -r requirements.txt
