# -*- coding: utf-8 -*-
"""
Created on Sat March 12 08:30:00 2022
@author: TVR Raviteja
"""

import pickle

import nltk
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy import stats

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Adding Subjectivity and Polarity columns
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))


def getsubj(text):
    return TextBlob(text).sentiment.subjectivity


def getpolarity(text):
    return TextBlob(text).sentiment.polarity


from PIL import Image

pickle_in = open("model_amzn.pkl", "rb")
amazon = pickle.load(pickle_in)

pickle_in = open("model_apl.pkl", "rb")
apple = pickle.load(pickle_in)

pickle_in = open("model_csco.pkl", "rb")
cisco = pickle.load(pickle_in)

pickle_in = open("model_fb.pkl", "rb")
facebook = pickle.load(pickle_in)

pickle_in = open("model_msft.pkl", "rb")
microsoft = pickle.load(pickle_in)

pickle_in = open("model_qcom.pkl", "rb")
qualcom = pickle.load(pickle_in)

pickle_in = open("model_tsla.pkl", "rb")
tesla = pickle.load(pickle_in)


def welcome():
    return "Welcome All"


# Functions to predict Close
import string


def char_rmvl(text):  # Removing all char except a-z and A-Z and replace them with ' '
    new = [char for char in text if char not in string.punctuation]
    new_str = ''.join(new)
    new.clear()
    return new_str


stop = stopwords.words('english')

# Apply lemmatization
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer  # used to perform lemmatization
from nltk.tokenize import word_tokenize


def lemmat(text):
    lemma = WordNetLemmatizer()
    words = word_tokenize(text)
    return ' '.join([lemma.lemmatize(word) for word in words])


sid = SentimentIntensityAnalyzer()


def filter(Headlines):
    Head = [x.lower() for x in Headlines]
    Head = [char_rmvl(x) for x in Head]
    Head = [' '.join([word for word in s.split() if word not in (stop)]) for s in Head]
    Head = [lemmat(s) for s in Head]
    return Head


def sent_anls(Head):
    compound = [sid.polarity_scores(x)['compound'] for x in Head]
    negative = [sid.polarity_scores(x)['neg'] for x in Head]
    neutral = [sid.polarity_scores(x)['neu'] for x in Head]
    positive = [sid.polarity_scores(x)['pos'] for x in Head]
    subjectivity = [getsubj(x) for x in Head]
    polarity = [getpolarity(x) for x in Head]

    return compound, negative, neutral, positive, subjectivity, polarity


day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)
nextday = '2021-06-24'
page_bg_img = ''

logo = Image.open('logo1.png')


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://wallpaperaccess.com/full/1393758.jpg");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


# eba205--Yellow
# 00C957--Green
# FF4040-- Red
def main():
    st.title("Stock Price Prediction Of Large Cap Tech Companies")

    st.image(logo)

    html_temp = """
    <div style="background-color:#00C957;padding:10px">
    <h2 style="color:White;text-align:center;">Close Price Predictor</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    set_bg_hack_url()

    option = st.selectbox('Choose a Company',
                          ('Amazon', 'Apple', 'Cisco', 'Facebook', 'Microsoft', 'Qualcom', 'Tesla'))

    if option == 'Amazon':
        linear = amazon
        image1 = Image.open('amzn_30_day.png')
        image2 = Image.open('amzn_30_day_join.png')

    elif option == 'Apple':
        linear = apple
        image1 = Image.open('apl_30_day.png')
        image2 = Image.open('apl_30_day_join.png')

    elif option == 'Cisco':
        linear = cisco
        image1 = Image.open('csco_30_day.png')
        image2 = Image.open('csco_30_day_join.png')

    elif option == 'Facebook':
        linear = facebook
        image1 = Image.open('fb_30_day.png')
        image2 = Image.open('fb_30_day_join.png')

    elif option == 'Microsoft':
        linear = microsoft
        image1 = Image.open('msft_30_day.png')
        image2 = Image.open('msft_30_day_join.png')

    elif option == 'Qualcom':
        linear = qualcom
        image1 = Image.open('qcom_30_day.png')
        image2 = Image.open('qcom_30_day_join.png')

    else:
        linear = tesla
        image1 = Image.open('tsla_30_day.png')
        image2 = Image.open('tsla_30_day_join.png')

    Open = st.text_input("Open", "Enter a value")
    High = st.text_input("High", "Enter a value")
    Low = st.text_input("Low", "Enter a value")
    Volume = st.text_input("Volume", "Enter a value")
    Headlines = st.text_input("Headlines", "Enter Recent Headlines")
    Headlines = list(Headlines.split("-"))
    head = filter(Headlines)
    cmpd, negt, neut, post, subj, pol = sent_anls(head)

    af = pd.DataFrame()
    af['compound'] = cmpd
    af['negative'] = negt
    af['neutral'] = neut
    af['positive'] = post
    af['Open'] = Open
    af['High'] = High
    af['Low'] = Low
    af['Volume'] = Volume
    af['Subjectivity'] = subj
    af['Polarity'] = pol

    result = ""

    if st.button("Predict"):
        result = linear.predict(af)[0]
        st.image(image1)
        st.text("Next 30 days Trend(Orange curve) from " + nextday)
        st.image(image2)
        st.text("Next 30 days trend from " + nextday + " along with previous 100 days trend")
    st.success('Predicted Close Price : $ {}'.format(result))

    if st.button("About"):
        st.text(
            "Close price of a stock is predicted based on news headlines and historical data using Machine Learning and Deep Learning Techniques")


if __name__ == '__main__':
    main()