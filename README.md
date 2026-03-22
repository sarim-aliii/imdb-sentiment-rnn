# 🎬 ReelFeel: IMDB Movie Review Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

**[✨ Try the Live Demo Here! ✨](https://imdb-movie-review-analysis-rnn.streamlit.app/)**

ReelFeel is a deep learning web application that classifies movie reviews as **Positive** or **Negative**. Built using a **Recurrent Neural Network (RNN)** and deployed with **Streamlit**, this project demonstrates end-to-end NLP preprocessing, model inference, and interactive data visualization.

---

## 🚀 Features
* **Real-time Prediction:** Enter any movie review and get an instant sentiment score.
* **Deep Learning Backend:** Utilizes an RNN trained on the classic IMDB dataset (50,000 reviews).
* **Word Cloud Generation:** Automatically visualizes the most impactful keywords from your input.
* **Automated Preprocessing:** Handles text tokenization and sequence padding internally to match model requirements.
* **Interactive UI:** A clean, minimalist, and color-coded interface powered by Streamlit.

---

## 🧠 Model Architecture
The core of this project is a **Simple RNN** built with TensorFlow/Keras:
* **Embedding Layer:** Maps words into a 128-dimensional vector space.
* **SimpleRNN Layer:** Processes sequences of up to 500 words.
* **Dense Layer:** A single neuron with a Sigmoid activation function to output a probability score.

$$\text{Sentiment} = \begin{cases} \text{Positive} & \text{if } p > 0.5 \\ \text{Negative} & \text{if } p \leq 0.5 \end{cases}$$

---

## 🛠️ Installation & Setup

To run this project locally on your machine:

1. **Clone the repository:**
   git clone [https://github.com/sarim-aliii/imdb-sentiment-rnn.git](https://github.com/sarim-aliii/imdb-sentiment-rnn.git)
   cd imdb-sentiment-rnn

2. **Install the required dependencies:**

pip install -r requirements.txt

3. **Run the Streamlit app:**

streamlit run app.py