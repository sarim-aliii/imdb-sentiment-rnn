import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# --- Page Configuration ---
st.set_page_config(page_title="ReelFeel AI", page_icon="🎬", layout="centered")

# --- Custom Styling ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model & Data ---
@st.cache_resource
def load_my_model():
    model = load_model('simple_rnn_imdb.h5')
    word_index = imdb.get_word_index()
    return model, word_index

model, word_index = load_my_model()

# --- Helper Functions ---
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    words = text.split()

    encoded_review = []
    for word in words:
        if word in word_index:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2)
            
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# --- UI Header ---
st.title("🎬 ReelFeel AI")
st.write("Analyze the sentiment of your movie reviews using Deep Learning.")

# --- User Input ---
user_input = st.text_area('Paste your review below:', height=150)

if st.button('Analyze Sentiment'):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner('Calculating sentiment...'):
            # 1. Prediction Logic
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)[0][0]
            sentiment = 'Positive' if prediction > 0.5 else 'Negative'
            
            # 2. UI Colors
            bg_color = "#d4edda" if sentiment == 'Positive' else "#f8d7da"
            txt_color = "#155724" if sentiment == 'Positive' else "#721c24"

            # 3. Display Result
            st.markdown(f'<div class="sentiment-box" style="background-color: {bg_color}; color: {txt_color};">'
                        f'{sentiment} Review ({prediction:.2%})</div>', unsafe_allow_html=True)

            # 4. Word Cloud Generation
            st.subheader("🔠 Review Keywords")
            
            # Create the wordcloud object
            wc = WordCloud(
                background_color='white' if sentiment == 'Positive' else '#1a1a1a',
                colormap='viridis' if sentiment == 'Positive' else 'magma',
                width=800, 
                height=400
            ).generate(user_input)

            # Display via Matplotlib
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            # 5. Final Verdict
            if prediction > 0.9:
                st.balloons()
                st.success("This review is glowing!")
            elif prediction < 0.1:
                st.error("This review is quite harsh.")