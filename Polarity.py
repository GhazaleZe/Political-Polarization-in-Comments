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
trump_df['Candidate'] = 'Trump'
biden_df['Candidate'] = 'Biden'

data_mixed = pd.concat([trump_df, biden_df])
data_mixed.sort_values(by='created_at')
data_mixed['country'] = data_mixed['country'].replace({"United States of America": "US", "United States": "US"})

## Cleaning the data
lm = WordNetLemmatizer()


def clean(text):
    text = str(text).lower()
    text = re.sub('[^a-z]', ' ', str(text))
    text = text.split()
    text = [lm.lemmatize(j) for j in text if j not in set(stopwords.words('english'))]
    text = ' '.join(str(k) for k in text)
    return text


def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getpolarity(text):
    return TextBlob(text).sentiment.polarity


def getAnalysis(score):
    if score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'


# Candidate Trump analysis

trump_tweets = \
    data_mixed.query('(Candidate == "Trump")').sort_values('user_followers_count', ascending=False).drop_duplicates(
        ['user_name'])[['user_id', 'tweet', 'country']]
trump_tweets = trump_tweets.dropna().loc[trump_tweets.country == 'US']
trump_tweets.reset_index(inplace=True, drop=True)

trump_tweets['cleantext'] = trump_tweets['tweet'].apply(clean)
# print("Trump Tweets", trump_tweets.head(5))
# %%
trump_tweets['subjectivity'] = trump_tweets['cleantext'].apply(getsubjectivity)
# %%
trump_tweets['polarity'] = trump_tweets['cleantext'].apply(getpolarity)
# %%
trump_tweets['analysis'] = trump_tweets['polarity'].apply(getAnalysis)
print("This is trump tweets analysis", trump_tweets.analysis.value_counts(normalize=True) * 100)

# Candidate Biden Analysis
biden_tweets = data_mixed.query('(Candidate == "Biden")').sort_values('user_followers_count',ascending=False).drop_duplicates(['user_name'])[['user_id','tweet','country']]
biden_tweets = biden_tweets.dropna().loc[biden_tweets.country =='US']
biden_tweets.reset_index(inplace = True ,drop =True)
biden_tweets['cleantext'] = biden_tweets['tweet'].apply(clean)

biden_tweets['subjectivity'] = biden_tweets['cleantext'].apply(getsubjectivity)
biden_tweets['polarity'] = biden_tweets['cleantext'].apply(getpolarity)
biden_tweets['analysis'] = biden_tweets['polarity'].apply(getAnalysis)

print("This is Biden tweets analysis", biden_tweets.analysis.value_counts(normalize=True) * 100)