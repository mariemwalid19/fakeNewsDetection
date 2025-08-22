import pickle
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# upload the data

model = pickle.load(open('E:/fake_news_model.sav', 'rb'))

with open('E:/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

st.title("Fake News Detection Web App")
st.info("Application For Fake News Detection")

News = st.text_input('Enter the news content here:')

st.sidebar.header('Press Predict')
btn = st.sidebar.button('Predict')

if btn:
    sequences = tokenizer.texts_to_sequences([News])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

    result = model.predict(padded_sequences)[0][0]

    if result < 0.5:
        st.sidebar.write('Real News')
        st.sidebar.image('https://th.bing.com/th/id/OIP.6ljulNuCMspTyI66pFAwcQHaHa?w=179&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7', width=150)
    else:
        st.sidebar.write("Fake News")
        st.sidebar.image('https://ichef.bbci.co.uk/images/ic/1024x576/p05vg83c.jpg', width=150)