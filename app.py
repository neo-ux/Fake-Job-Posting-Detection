import streamlit as st
import joblib

model = joblib.load("fake_job_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("🕵️‍♂️ Fake Job Posting Detector")
user_input = st.text_area("Paste the job description here:")

if st.button("Predict"):
    if user_input.strip():
        X_input = tfidf.transform([user_input])
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][pred]

        if pred == 1:
            st.error(f"⚠️ Fake Job Detected! ({prob * 100:.2f}% confidence)")
        else:
            st.success(f"✅ Looks like a Real Job! ({prob * 100:.2f}% confidence)")
    else:
        st.warning("Please enter some job description text.")
