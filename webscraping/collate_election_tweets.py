# Brandon 1505211
# from datetime import datetime
# import GetOldTweets3 as got
# from . import GetOldTweets3 as got
# import GetOldTweets3.manager as tm
# from GetOldTweets3.manager import TweetCriteria, GitManager
# import GetOldTweets3.manager.TweetCriteria
# import GetOldTweets3.manager.GitManager
# from GetOldTweets3.manager.TweetCriteria import TweetCriteria
from tools.GetOldTweets3.manager import TweetCriteria, GitManager
import pandas as pd
from datetime import timedelta, date
from tools.twitter_preprocessor import TwitterPreprocessor
from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
from tqdm.auto import tqdm
import time


# https://en.wikipedia.org/wiki/Centre_points_of_the_United_Kingdom

# def multithread():
#     start_date = date(2019, 12, 5)
#     end_date = date(2019, 12, 18)
#     started_jobs = []
#     completed_jobs = []
#     with ThreadPoolExecutor(max_workers=5) as executor:
#         pbar = tqdm(total=14)
#         while start_date <= end_date:
#             print("\n\nStart date: ", str(start_date))
#             # executor.submit(main, start_date)
#             started_jobs.append(executor.submit(get_tweets, start_date))
#             pbar.update(1)
#             start_date += timedelta(days=1)
#             time.sleep(30)


def get_tweets(start_date):
    end_date = start_date + timedelta(days=1)
    # tm = GetOldTweets3.manager
    # tm = got.manager
    # tc = TweetCriteria.TweetCriteria
    # tc.TweetCriteria.setQuerySearch()
    tm = GitManager.TweetManager()
    tc = TweetCriteria.TweetCriteria()
    criteria = tc.setQuerySearch(querySearch='general election') \
        .setSince(str(start_date)) \
        .setUntil(str(end_date)) \
        .setNear("Dunsop Bridge, Lancashire") \
        .setWithin("485mi") \
        .setMaxTweets(15000)

    tweets = tm.getTweets(criteria)
    return tweets


def main():
    # my_tuple = preprocess_collated_tweets()
    start_date = date(2019, 12, 6)
    end_date = date(2019, 12, 17)
    started_jobs = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        pbar = tqdm(total=12)
        for day in pd.date_range(start_date, end_date):
            day = pd.to_datetime(day).date()

            started_jobs.append((executor.submit(get_tweets, day), day))

        completed_jobs = [(job.result(), day) for job, day in started_jobs]

        for job, day in completed_jobs:
            print("\nCompleted start date: ", day)
            count = 0

            preprocessed_tweets = []
            hashtags = []
            t_date = []
            my_tuple = (preprocessed_tweets, hashtags, t_date)
            for tweet in job:
                # count += 1
                # print(count, tweet.text)
                # preprocessed_tweets.append(preprocess(tweet.text))
                my_tuple[0].append(preprocess(tweet.text))
                my_tuple[1].append(tweet.hashtags)
                my_tuple[2].append(tweet.date)

            preprocessed_df = pd.DataFrame(my_tuple[0], columns=['text'])
            preprocessed_df['hashtags'] = my_tuple[1]
            preprocessed_df['date'] = my_tuple[2]
            print("\nUnprocessed Tweets for" + str(day) + ": ")
            print(preprocessed_df.head())
            # time_format = datetime.now().strftime(' %m-%d-%y %H:%M')
            # preprocessed_df.to_csv('data/collated_election_tweets{}'.format(time_format) + '.data', encoding='utf-8',
            #                        index=False)

            preprocessed_df.to_csv(
                'data/max/collated_election_tweets{}'.format(' ' + str(day)) + '.csv',
                encoding='utf-8',
                index=False)
            time.sleep(30)
            pbar.update(1)
        pbar.close()


# def preprocess_collated_tweets():
#     count = 0
#     print("\nUnprocessed Tweets: ")
#     preprocessed_tweets = []
#     hashtags = []
#     date = []
#     my_tuple = (preprocessed_tweets, hashtags, date)
#     for tweet in get_tweets():
#         count += 1
#         print(count, tweet.text)
#         # preprocessed_tweets.append(preprocess(tweet.text))
#         my_tuple[0].append(preprocess(tweet.text))
#         my_tuple[1].append(tweet.hashtags)
#         my_tuple[2].append(tweet.date)
#
#     return my_tuple


def preprocess(tweet):
    preprocessed = TwitterPreprocessor(tweet).fully_preprocess()
    return preprocessed.text


if __name__ == '__main__':
    main()
