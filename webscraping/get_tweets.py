from datetime import timedelta, date
import pandas as pd
import time
from tqdm import tqdm

from tools.twitter_preprocessor import TwitterPreprocessor


# time_format = "2019-12-05"


# def date_range():
#     start_date = date(2019, 12, 5)
#     end_date = date(2019, 12, 18)
#     pbar = tqdm(total=14)
#     while start_date <= end_date:
#         # preprocess_collated_tweets(start_date)
#         main(str(start_date))
#         pbar.update(1)
#         start_date += timedelta(days=1)
#         time.sleep(900)


def get_tweets(start_date):
    tm = tools.GetOldTweets3.manager
    criteria = tools.GetOldTweets3.manager.TweetCriteria().setQuerySearch('general election') \
        .setSince(start_date) \
        .setUntil(start_date) \
        .setNear("Dunsop Bridge, Lancashire") \
        .setWithin("480mi") \
        .setMaxTweets(10)
    # .setTopTweetsx/(True)

    # try 5000 tweets per day

    tweets = tm.TweetManager.getTweets(criteria)
    return tweets


def main():
    start_date = date(2019, 12, 5)
    end_date = date(2019, 12, 18)
    pbar = tqdm(total=14)
    while start_date <= end_date:
        my_tuple = preprocess_collated_tweets(str(start_date))
        preprocessed_df = pd.DataFrame(my_tuple[0], columns=['text'])
        preprocessed_df['hashtags'] = my_tuple[1]
        preprocessed_df['date'] = my_tuple[2]
        preprocessed_df.to_csv('data/collated_election_tweets{}'.format(str(start_date)) + '.csv', encoding='utf-8',
                               index=False)
        pbar.update(1)
        start_date += timedelta(days=1)
        time.sleep(60)


def preprocess_collated_tweets(start_date):
    count = 0
    # print("\nUnprocessed Tweets: ")
    preprocessed_tweets = []
    hashtags = []
    t_date = []
    my_tuple = (preprocessed_tweets, hashtags, t_date)
    for tweet in get_tweets(start_date):
        count += 1
        print(count, tweet.text)
        # preprocessed_tweets.append(preprocess(tweet.text))
        my_tuple[0].append(preprocess(tweet.text))
        my_tuple[1].append(tweet.hashtags)
        my_tuple[2].append(tweet.t_date)
        # time.sleep(0.01)

    return my_tuple


def preprocess(tweet):
    preprocessed = TwitterPreprocessor(tweet).fully_preprocess()
    return preprocessed.text


if __name__ == '__main__':
    # date_range()
    main()
