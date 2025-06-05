import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


df = pd.read_csv("/Users/vaibhavanand/Downloads/fake_job_postings.csv")


df.drop(columns=["job_id", "salary_range", "depasrtment"], inplace=True)


df["text_data"] = (
    df["title"].fillna("") + " " +
    df["company_profile"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["requirements"].fillna("") + " " +
    df["benefits"].fillna("")
)


df.drop(columns=["title", "company_profile", "description", "requirements", "benefits"], inplace=True)
df = df[df["text_data"].str.strip() != ""]


for col in ['location', 'employment_type', 'required_experience', 'required_education', 'industry', 'function']:
    df[col] = df[col].fillna("Unknown")

X_text = df["text_data"]
y = df["fraudulent"]
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(X_text)


X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


joblib.dump(model, "fake_job_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
