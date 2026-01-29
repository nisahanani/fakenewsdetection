import streamlit as st
import pickle
import re

# =========================
# Load pipeline model
# =========================
with open("model_news.pkl(1)", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection System")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

user_input = st.text_area("Enter news text:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        prediction = model.predict([cleaned_text])[0]

        if prediction == 1:
            st.success("🟢 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")
