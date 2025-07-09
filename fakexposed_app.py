import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fakexposed", page_icon="🕵️", layout="centered")

# Custom CSS for background image and styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-image: url('https://images.unsplash.com/photo-1505685296765-3a2736de412f?auto=format&fit=crop&w=1400&q=80');
        background-size: cover;
        background-attachment: fixed;
        color: #1f2937;
    }
    .main-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 0 30px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    .title {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    .result {
        text-align: center;
        font-size: 1.5rem;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        margin-top: 2rem;
    }
    .fake {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    .real {
        background-color: #d1fae5;
        color: #065f46;
    }
    </style>
""", unsafe_allow_html=True)

# Layout structure
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="title">Fakexposed 🕵️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Check if a news article is Real or Fake using AI</div>', unsafe_allow_html=True)

# User input
user_input = st.text_area("Paste a news article or headline here:")

# Predict
if st.button("🚨 Detect Now"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        input_vec = vectorizer.transform([user_input])
        pred = model.predict(input_vec)[0]
        conf = model.predict_proba(input_vec)[0][pred]
        if pred == 0:
            st.markdown(f'<div class="result fake">🔴 Fake News Detected!<br>Confidence: {conf:.2%}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result real">🟢 Real News Detected!<br>Confidence: {conf:.2%}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

