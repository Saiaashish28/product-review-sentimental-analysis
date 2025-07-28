import streamlit as st
import joblib
import os

# --- Page Config ---
st.set_page_config(page_title="Product Review Sentiment Analyzer", page_icon="🛍️", layout="centered")

# --- Title and Description ---
st.title("🛍️ Product Review Sentiment Analyzer")
st.markdown("Analyze product reviews to determine whether they're **Positive** or **Negative**.")

st.divider()

# --- Load Model and Vectorizer ---
model_path = "sentiment_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model or vectorizer file not found. Please make sure `sentiment_model.pkl` and `tfidf_vectorizer.pkl` are in the current directory.")
    st.stop()

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# --- Text Input ---
st.subheader("📝 Enter a Review")
review = st.text_area("Write your product review here:", height=150)

# --- Prediction ---
if st.button("🔍 Analyze"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review before analyzing.")
    else:
        # Transform and predict
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]

        sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
        sentiment_color = "green" if prediction == 1 else "red"

        # Optional: Confidence (if model supports predict_proba)
        try:
            prob = model.predict_proba(review_vector)[0]
            confidence = round(max(prob) * 100, 2)
            st.markdown(f"<h4 style='color:{sentiment_color}'>Sentiment: {sentiment} ({confidence}% confidence)</h4>", unsafe_allow_html=True)
        except:
            st.markdown(f"<h4 style='color:{sentiment_color}'>Sentiment: {sentiment}</h4>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Your friendly review analyzer bot.")
