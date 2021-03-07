# Brandon 1505211
# -*- coding: utf-8 -*-
from datetime import timedelta, date
import pandas as pd
from tools.twitter_preprocessor import TwitterPreprocessor
from tqdm import tqdm
import time
import concurrent.futures


def get_tweets(start_date):
    # start_date = str(date(2019, 12, 5))
    end_date = start_date + timedelta(days=1)

    tm = tools.GetOldTweets3.manager
    start_date = "2019-12-05"
    criteria = tools.GetOldTweets3.manager.TweetCriteria().setQuerySearch('general election') \
        .setSince(str(start_date)) \
        .setUntil(str(end_date)) \
        .setNear("Dunsop Bridge, Lancashire") \
        .setWithin("480mi") \
        .setMaxTweets(25000)
    # .setTopTweetsx/(True)

    tweets = tm.TweetManager.getTweets(criteria)
    # print(tweets)
    return tweets


def main():
    start_date = date(2019, 12, 5)
    end_date = date(2019, 12, 18)
    pbar = tqdm(total=14)
    while start_date <= end_date:
        # print(str(start_date))
        # my_tuple = preprocess_collated_tweets(start_date)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # future = executor.submit(preprocess_collated_tweets, str(start_date))
        # my_tuple = future.result()

        count = 0
        print("\n\nUnprocessed Tweets: ")
        preprocessed_tweets = []
        hashtags = []
        t_date = []
        my_tuple = (preprocessed_tweets, hashtags, t_date)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # threaded execution
        fetch_tweets = executor.submit(get_tweets, start_date).result()

        for tweet in fetch_tweets:
            count += 1
            print(count, tweet.text)
            # preprocessed_tweets.append(preprocess(tweet.text))
            my_tuple[0].append(preprocess(tweet.text))
            my_tuple[1].append(tweet.hashtags)
            my_tuple[2].append(tweet.date)

        preprocessed_df = pd.DataFrame(my_tuple[0], columns=['text'])
        preprocessed_df['hashtags'] = my_tuple[1]
        preprocessed_df['date'] = my_tuple[2]
        preprocessed_df.to_csv('data/output/collated_election_tweets{}'.format(' ' + str(start_date)) + '.csv', encoding='utf-8',
                               index=False)

        start_date += timedelta(days=1)
        pbar.update(1)

        time.sleep(900)  # sleep for 15mins


def preprocess(tweet):
    preprocessed = TwitterPreprocessor(tweet).fully_preprocess()
    return preprocessed.text


if __name__ == '__main__':
    main()
