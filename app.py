from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch.nn.functional as F
import streamlit as st
import tensorflow as tf
import yfinance as yf
import numpy as np
import datetime
import requests
from joblib import load
from bs4 import BeautifulSoup
from newsapi import NewsApiClient

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    return model, tokenizer

def get_news_data(t):
    newsapi = NewsApiClient(api_key=st.secrets["newsapi"])

    all_articles = newsapi.get_everything(q=f'{t.info["symbol"]} OR {t.info["shortName"]}',
                                          domains='forbes.com,marketwatch.com',
                                          from_param=str(datetime.date.today() - datetime.timedelta(5)),
                                          language='en',
                                          sort_by='relevancy')['articles'][:10]
    texts = []
    for article in all_articles[:10]:
        response = requests.get(article['url'])
        html_content = response.text

        soup = BeautifulSoup(html_content, 'html.parser')

        p_tags = soup.find_all('p')
        text=''

        for tag in p_tags:
            text+=tag.get_text() + " "
      
        texts.append(text)
    return texts
        

@st.cache_resource
def get_model(_):
    model = tf.keras.models.load_model('models/lstm.hdf5')

    HORIZON = 1
    WINDOW = 7

    def get_windows(x, window_size=WINDOW, horizon=HORIZON):
        window_indexes = np.expand_dims(np.arange(len(x)-(horizon+window_size-1)), axis=1) + np.expand_dims(np.arange(horizon+window_size), axis=0)
        return x[window_indexes]
    
    data = yf.Ticker("^GSPC").history(interval='1d', period='15y', actions=False)['Close'].values
    
    windowed_dataset = get_windows(data)
    X, y = windowed_dataset[:, :-HORIZON], windowed_dataset[:, -HORIZON:]

    model.fit(X, y,
              epochs=50,
              verbose=1,
              batch_size=32)
    
    return model

st.header("PathosTrade")
st.markdown("Predicting Stock Price with Sentiment and Science")

ticker = st.text_input("Predict tomorrow's stock price of", placeholder='Enter any stock ticker')
button  = st.button("Submit")
if button:
    if not ticker:
        st.warning("Please enter any stock ticker")
    else:
        t = yf.Ticker(ticker)
        data = t.history(interval='1d', period='7d', actions=False)["Close"].values
        if not len(data):
            st.error(f"Invalid ticker: **{ticker}**")
        else:
            model = get_model(datetime.date.today())
            timeseries_pred = model.predict(np.array([data]))

            news_data = get_news_data(t)

            model, tokenizer = load_sentiment_model()

            probabilities = 0

            for text in news_data:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)

                logits = outputs.logits
            
                probabilities+=F.softmax(logits, dim=1)

            probabilities =  ((probabilities/len(news_data))[0]).tolist()
            loaded_model = load('models/linear_regression.joblib')

            # st.write(f"{[[data[-1], probabilities[2], probabilities[0], probabilities[1], timeseries_pred[0][0]]]}")
            final_pred = loaded_model.predict([[data[-1], probabilities[2], probabilities[0], probabilities[1], timeseries_pred[0][0]]])[0]
            st.markdown(f"##### Predicted Price for {t.info['shortName']} is `${final_pred:.4f}`")

        
