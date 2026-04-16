import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Page settings
st.set_page_config(page_title="Sentiment Analysis", page_icon="=¬", layout="centered")

# Title
st.title("=¬ Customer Review Sentiment Analysis")

st.write("Enter a customer review and the model will predict the sentiment.")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("This app predicts sentiment of customer reviews using Machine Learning.")
st.sidebar.write("Model Used: **Support Vector Machine (SVM)**")

# Input box
review = st.text_area(" Enter Customer Review")

if st.button("= Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)

        if prediction[0] == 0:
            st.error("=! Negative Review")
        elif prediction[0] == 1:
            st.info("= Neutral Review")
        else:
            st.success("=
 Positive Review")