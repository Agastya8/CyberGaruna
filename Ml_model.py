from sklearn import *
import numpy as np 
import pandas as panda
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
   
import tweepy
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from time import time
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
    


def predt():
    model=joblib.load('cb_sgd_final.sav') #path of classifier
    df= panda.read_csv("./livedata/real_time_tweets.csv") # path of real time tweets
    count_vect = pickle.load(open('count_vect', 'rb')) # path of count vectorizer

    df['text length'] = df['text'].apply(len)
    tweet=df.text
    stopwords = nltk.corpus.stopwords.words("english")
    #extending the stopwords to include other words used in twitter such as retweet(rt) etc.
    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)
    stemmer = PorterStemmer()
    def preprocess(tweet):  
    
    # removal of extra spaces
        regex_pat = re.compile(r'\s+')
        tweet_space = tweet.str.replace(regex_pat, ' ')

    # removal of @name[mention]
        regex_pat = re.compile(r'@[\w\-]+')
        tweet_name = tweet_space.str.replace(regex_pat, '')

    # removal of links[https://abc.com]
        giant_url_regex =  re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                                      '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        tweets = tweet_name.str.replace(giant_url_regex, '')
    
    # removal of punctuations and numbers
        punc_remove = tweets.str.replace("[^a-zA-Z]", " ")
    # remove whitespace with a single space
        newtweet=punc_remove.str.replace(r'\s+', ' ')
    # remove leading and trailing whitespace
        newtweet=newtweet.str.replace(r'^\s+|\s+?$','')
    # replace normal numbers with numbr
        newtweet=newtweet.str.replace(r'\d+(\.\d+)?','numbr')
    # removal of capitalization
        tweet_lower = newtweet.str.lower()
    
    # tokenizing
        tokenized_tweet = tweet_lower.apply(lambda x: x.split())
    
    # removal of stopwords
        tokenized_tweet=  tokenized_tweet.apply(lambda x: [item for item in x if item not in stopwords])
    
    # stemming of the tweets
        tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
    
        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
            tweets_p= tokenized_tweet
    
        return tweets_p

    processed_tweets = preprocess(tweet)

    df['processed_tweets'] = processed_tweets
    #vectorizing the tweets
    testing_data = count_vect.transform(tweet)
    #predecting the tweets
    y_preds = model.predict(testing_data)
    dframe = panda.DataFrame()
    dframe['tweets']= df['text']
    dframe['class']=y_preds
    tottw=dframe['tweets'].count
    #open the file to write the predicted tweets and clear it
    f = open("./livedata/classified_tweets.csv", "w")
    f.truncate()
    f.close()
    #write the predicted tweets to the file
    dframe.to_csv('./livedata/classified_tweets.csv', index=False)
    #count for predicted tweets
    countc=0
    vals=dframe['class']

    for i in vals:
        if i==1:
            countc=countc+1
    
    return countc
    
