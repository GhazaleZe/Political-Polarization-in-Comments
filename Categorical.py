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
import jenkspy

nltk.download('stopwords')
nltk.download('wordnet')
lm = WordNetLemmatizer()

biden_tweets = pd.read_csv("biden_US_cleaned.csv")
trump_tweets = pd.read_csv("trump_US_cleaned.csv")


def user_activity(dataframe):
    tweet_counts = dataframe.groupby('user_id').size()
    labels = tweet_counts.apply(lambda x: 'active' if x > 5 else 'less active')
    dataframe['user_activity'] = dataframe['user_id'].map(labels)
    return dataframe


biden_tweets_activity = user_activity(biden_tweets)
trump_tweets_activity = user_activity(trump_tweets)

biden_tweets_activity[biden_tweets_activity['user_activity'] == 'active'].head()

print("Biden active user",biden_tweets_activity[biden_tweets_activity['user_activity'] == 'active'].shape)
print("Biden less active user",biden_tweets_activity[biden_tweets_activity['user_activity'] == 'less active'].shape)

print("Trump active user",trump_tweets_activity[trump_tweets_activity['user_activity'] == 'active'].shape)
print("Trump less active user",trump_tweets_activity[trump_tweets_activity['user_activity'] == 'less active'].shape)

def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def getpolarity(text):
    return TextBlob(text).sentiment.polarity
def getAnalysis(score):
    if score <0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'


def sentiment_analysis(tweets):
    tweets['subjectivity'] = tweets['cleantext'].apply(getsubjectivity)

    tweets['polarity'] = tweets['cleantext'].apply(getpolarity)

    tweets['analysis'] = tweets['polarity'].apply(getAnalysis)
    return tweets

def check_and_convert(value):
    if isinstance(value, float):  # Check if the value is a float
        return str(value)         # Convert float to string
    else:
        return value              # Return value unchanged if not a float

# Apply the function to the DataFrame column
trump_tweets_activity['cleantext'] = trump_tweets_activity['cleantext'].apply(lambda x: check_and_convert(x))

trump_df = sentiment_analysis(trump_tweets_activity)

biden_df = sentiment_analysis(biden_tweets_activity)

sorted_biden = biden_df.sort_values(by=['likes', 'retweet_count'], ascending=False)

print("biden_df.analysis.value_counts()", biden_df.analysis.value_counts())

print("trump_df.analysis.value_counts()",trump_df.analysis.value_counts())

sorted_trump = trump_df.sort_values(by=['likes', 'retweet_count'], ascending=False)

num_bins = 5

breaks = jenkspy.jenks_breaks(sorted_biden['likes'], n_classes= num_bins)
print(breaks)
sorted_biden['likes_category'] = pd.cut(sorted_biden['likes'], bins=breaks, labels=range(1, num_bins+1), include_lowest=True)

breaks = jenkspy.jenks_breaks(sorted_biden['retweet_count'], n_classes= num_bins)
print(breaks)
sorted_biden['retweet_count_category'] = pd.cut(sorted_biden['retweet_count'], bins=breaks, labels=range(1, num_bins+1), include_lowest=True)

sorted_biden.to_csv('sorted_biden_categorical.csv', index=False)
