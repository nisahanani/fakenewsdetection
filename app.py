import streamlit as st
import pickle
import re

# =========================
# Load models & vectorizer
# =========================
with open("model_lr.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("model_nb.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# =========================
# Text preprocessing
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write(
    "This application classifies news articles as **Fake** or **Real** "
    "using Logistic Regression and Naive Bayes models."
)

st.markdown("---")

# Text input
user_input = st.text_area(
    "Enter a news article text:",
    height=200
)

# Model selection
model_choice = st.selectbox(
    "Select classification model:",
    ("Logistic Regression", "Naive Bayes")
)

# Prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a news article.")
    else:
        cleaned_text = clean_text(user_input)
        text_vector = vectorizer.transform([cleaned_text])

        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(text_vector)[0]
        else:
            prediction = nb_model.predict(text_vector)[0]

        if prediction == 1:
            st.success("🟢 Prediction: REAL NEWS")
        else:
            st.error("🔴 Prediction: FAKE NEWS")

st.markdown("---")
st.caption("Developed for NLP Fake News Detection Project")
