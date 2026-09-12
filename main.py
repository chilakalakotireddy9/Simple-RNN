## import the libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

## load the IMDB word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

## load the IMDB dataset word index

model = load_model(
    r"C:\Users\n2300\Desktop\ANN\simple_rnn_imdb.h5"
)

print("Model loaded successfully!")


## step-2 helper function
## function to decodes reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

## function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2) + 3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    
    prediction=model.predict(preprocessed_input)
    
    sentiment='positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]

import streamlit as st
## streamlit app

st.title('IMDB Movie Review Sentiment analysis')
st.write('enter a movie review to classify it as positive or negative.')

## user input
user_input=st.text_area("Movie Review")

if st.button('Classify'):
    
    preprocess_input=preprocess_text(user_input)
    
    ## make prediction
    prediction=model.predict(preprocess_input)
    sentiment='positive' if prediction[0][0] > 0.5 else 'negative'
    
    ## diaplay the result
    
    st.write(f"sentiment:{sentiment}")
    st.write(f"prediction score:{prediction[0][0]}")

else:
    st.write("please enter the movie review.")



    
