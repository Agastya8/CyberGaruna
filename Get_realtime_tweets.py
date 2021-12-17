

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time
import json
import csv
import pandas as pd
consumer_key="" #insert your consumer key
consumer_secret="" #insert your consumer secret
access_token_key="" #insert your access token key
access_token_secret="" #insert your access token secret


class TwitterAuthenticator():

    def authenticate_twitter_app(self):        
        auth=OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token_key, access_token_secret)
        return auth

class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()    

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app() 
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(track=hash_tag_list)

class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    #listen tweets for a certain time (3 mins ie 180secs)
    def __init__(self, fetched_tweets_filename, time_limit=180):
        self.start_time = time.time()
        self.limit = time_limit
        self.fetched_tweets_filename = fetched_tweets_filename
        super(TwitterListener, self).__init__()


    def on_data(self, data):
        if (time.time() - self.start_time) < self.limit:
            try:
                with open(self.fetched_tweets_filename, 'a') as tf:
                    tf.write(data)
                return True
            except BaseException as e:
                print("Error on_data %s" % str(e))
            return True
        else:
          return False
          
    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)


if __name__ == '__main__':
    hash_tag_list = ["you're dumb", "you're ugly", "youre pussy", "fuck you", "bitch"]
    
    #file name to beaved as json
    fetched_tweets_filename = "./livedata/real_time_tweets.json"
    twitter_streamer = TwitterStreamer()
    #open the json file and write real time twteets
    open('./livedata/real_time_tweets.json', 'w').close()
    print("Started...")
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)
    
    #ope the written jason file and convert it to dataframe
    df = pd.read_json("./livedata/real_time_tweets.json", lines=True)
    
    #clean the tweets of last time
    f = open("./livedata/real_time_tweets.csv", "w")
    f.truncate()
    f.close()
    
    #convert dataframe to csv
    df.to_csv("./livedata/real_time_tweets.csv")
    print("Done!")





