import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from wordcloud import wordcloud
from textblob import TextBlob
# from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.sentiment.util import *

nltk.download('stopwords')
nltk.download('wordnet')
biden_df = pd.read_csv(r"Data/hashtag_joebiden.csv", lineterminator='\n')
trump_df = pd.read_csv(r"Data/hashtag_donaldtrump.csv", lineterminator='\n')
lm = WordNetLemmatizer()


def clean(text):
    text = str(text).lower()
    text = re.sub('[^a-z]', ' ', str(text))
    text = text.split()
    text = [lm.lemmatize(j) for j in text if j not in set(stopwords.words('english'))]
    text = ' '.join(str(k) for k in text)
    return text


def ready_data(candidate_df):
    candidate_df.sort_values(by='created_at')
    candidate_df['country'] = candidate_df['country'].replace({"United States of America": "US", "United States": "US"})
    print(candidate_df.shape)
    candidate_df = \
        candidate_df.sort_values('user_followers_count', ascending=False).drop_duplicates(['user_name', 'tweet'])[
            ['user_id', 'tweet', 'country', 'likes', 'retweet_count']]
    print(candidate_df.shape)
    candidate_df = candidate_df.dropna().loc[candidate_df.country == 'US']
    print(candidate_df.shape)
    candidate_df.reset_index(inplace=True, drop=True)

    candidate_df['cleantext'] = candidate_df['tweet'].apply(clean)
    candidate_df.drop(['tweet', 'country'], axis=1, inplace=True)
    print(candidate_df.shape)
    return candidate_df


biden_tweets = ready_data(biden_df)
print(biden_tweets.shape)
biden_tweets.to_csv('biden_US_cleaned.csv', index=False)
trump_tweets = ready_data(trump_df)
print(trump_tweets.shape)
trump_tweets.to_csv('trump_US_cleaned.csv', index=False)
print("Data is clean and saved.")