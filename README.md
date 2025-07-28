# 📝 Product Review Sentiment

A web app that analyzes product reviews and classifies them as **Positive** or **Negative** using machine learning.

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Streamlit-red?logo=streamlit">
  <img src="https://img.shields.io/badge/Model-Logistic%20Regression-blue">
  <img src="https://img.shields.io/badge/NLP-TFIDF-orange">
</p>

---

## 🔍 Features

- 🔠 Predicts sentiment of product reviews as **Positive** or **Negative**
- 📈 Displays confidence score for predictions
- ✍️ Supports manual text input and `.txt` file upload
- ✅ Clean and responsive user interface
- 💾 Built with a lightweight logistic regression model and TF-IDF vectorizer

---

## 📸 Screenshots

<img width="1047" height="714" alt="image" src="https://github.com/user-attachments/assets/24195e39-6adc-490a-9c86-768169e2b925" />

---

## 🧠 How It Works

- Reviews are preprocessed and labeled (positive/negative).
- A TF-IDF vectorizer converts text into feature vectors.
- A logistic regression model classifies the review sentiment.
- The trained model is saved using `joblib` and loaded in the Streamlit app.


---

## 📦 Installation

1. **Clone the repo:**
```bash
git clone https://github.com/Saiaashish28/product-review-sentimental-analysis.git
cd product-review-sentiment


