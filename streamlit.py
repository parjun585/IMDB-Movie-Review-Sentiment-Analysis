import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('simpleRNN_IMDB_sentiment Analysis_part 2.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the word index dictionary for IMDB dataset
word_index = tf.keras.datasets.imdb.get_word_index()

# Set maximum length for padding
max_length = 500

# Function to preprocess text input for prediction
def preprocess_text(text):
    # Convert the text to lowercase and split it into words
    words = text.lower().split()
    # Encode the words into integers using the word index
    encoded_text = [word_index.get(word, 2) + 3 for word in words]  # +3 because IMDB indices are offset
    # Pad the sequence to match the training input size
    padded_text = tf.keras.preprocessing.sequence.pad_sequences([encoded_text], maxlen=max_length)
    return padded_text

# Function to predict the sentiment of a given review
def predict_sentiment(review):
    # Preprocess the review for model prediction
    preprocessed_input = preprocess_text(review)
    # Predict sentiment using the pre-trained model
    prediction = model.predict(preprocessed_input)
    # Determine the sentiment based on prediction score
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review, and the model will predict whether the sentiment is positive or negative.")

# User input for movie review
review_input = st.text_area("Enter your review:", height=200)

# If the user clicks the 'Analyze' button
if st.button("Analyze Sentiment"):
    if review_input.strip() != "":
        sentiment, score = predict_sentiment(review_input)
        # Display the results
        st.write(f"**Sentiment**: {sentiment}")
        st.write(f"**Confidence Score**: {score:.4f}")
    else:
        st.write("Please enter a review to analyze.")

# Footer
st.write("This app uses a pre-trained SimpleRNN model to analyze the sentiment of movie reviews from the IMDB dataset.")


