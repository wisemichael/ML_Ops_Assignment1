import streamlit as st
import joblib

# App title and description
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and this app will predict whether it's Positive or Negative.")

# Load the saved model
@st.cache_data
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()



# User input section
st.subheader("Enter a Movie Review to Analyze")
user_input = st.text_area("Movie Review", "")

# Analyze button
if st.button("Analyze", key="analyze_button"):
    if user_input.strip() == "":
        st.warning("Please enter a review before analyzing.")
    else:
        prediction = model.predict([user_input])[0]
        label = "Positive" if prediction == 1 else "Negative"
        st.success(f"Predicted Sentiment: **{label}**")

if st.button("Analyze"):
    if user_input:  # Make sure user typed something
        # Predict sentiment
        prediction = model.predict([user_input])[0]  # Predict expects a list
        
        # Predict probability
        probability = model.predict_proba([user_input])[0].max()
        
        # Show result
        if prediction == 1:
            st.success(f"Predicted Sentiment: Positive 👍 (Confidence: {probability:.2f})")
        else:
            st.error(f"Predicted Sentiment: Negative 👎 (Confidence: {probability:.2f})")
    else:
        st.warning("Please enter a movie review before analyzing.")

