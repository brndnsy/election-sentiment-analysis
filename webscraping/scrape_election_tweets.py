import datetime as dt
from datetime import datetime
from twitterscraper import query_tweets
import pandas as pd

from tools.twitter_preprocessor import TwitterPreprocessor


def get_tweets():
    tweets = query_tweets(query="General Election",
                          begindate=dt.date(2019, 12, 5),
                          enddate=dt.date(2019, 12, 19),
                          limit=300000,
                          lang="en",
                          poolsize=300)

    # print the retrieved tweets to the screen:
    collated_tweets = []
    for tweet in tweets:
        collated_tweets.append(preprocess(tweet.text))

        # Or save the retrieved tweets to file:
        # file = open(“output.txt”,”w”)
        # for tweet in query_tweets("Trump OR Clinton", 10):
        #     file.write(tweet.encode('utf-8'))
        # file.close()
    df = pd.DataFrame(collated_tweets, columns=['text'])
    time_format = datetime.now().strftime(' %m-%d-%y %H:%M')
    df.to_csv('./data/collated_election_tweets{}'.format(time_format) + '.csv', encoding='utf-8',
              index=False)

    return df


def main():
    get_tweets()


def preprocess(tweet):
    preprocessed = TwitterPreprocessor(tweet).fully_preprocess()
    return preprocessed.text


if __name__ == '__main__':
    main()
