# 🎬 ReelFeel: IMDB Movie Review Sentiment Analysis

ReelFeel is a deep learning web application that classifies movie reviews as **Positive** or **Negative**. Built using a **Recurrent Neural Network (RNN)** and deployed with **Streamlit**, this project demonstrates end-to-end NLP preprocessing and model inference.

---

## 🚀 Features
* **Real-time Prediction:** Enter any movie review and get an instant sentiment score.
* **Deep Learning Backend:** Utilizes an RNN trained on the classic IMDB dataset (50,000 reviews).
* **Automated Preprocessing:** Handles text tokenization and sequence padding internally to match model requirements.
* **Interactive UI:** A clean, minimalist interface powered by Streamlit.

---

## 🧠 Model Architecture
The core of this project is a **Simple RNN** built with TensorFlow/Keras:
* **Embedding Layer:** Maps words into a 128-dimensional vector space.
* **SimpleRNN Layer:** Processes sequences of up to 500 words.
* **Dense Layer:** A single neuron with a Sigmoid activation function to output a probability score.

$$\text{Sentiment} = \begin{cases} \text{Positive} & \text{if } p > 0.5 \\ \text{Negative} & \text{if } p \leq 0.5 \end{cases}$$

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sarim-aliii/imdb-sentiment-rnn.git](https://github.com/sarim-aliii/imdb-sentiment-rnn.git)
   cd imdb-sentiment-rnn
