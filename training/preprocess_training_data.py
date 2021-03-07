# http://help.sentiment140.com/for-students/
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from tools.twitter_preprocessor import TwitterPreprocessor


# function to import Stanford's training dataset
def import_dataset():
    cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
    df = pd.read_csv("../data/training.1600000.processed.noemoticon.csv", header=None, names=cols, encoding='latin-1')
    # df['sentiment'] = df['sentiment'].map(int)  # convert string to integer
    df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
    df.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)  # remove unnecessary columns
    return df


def preprocess(tweet):  # preprocesses tweets by removing irrelevant info e.g. mentions, hashtags, urls etc
    preprocessed = TwitterPreprocessor(tweet)
    return preprocessed.fully_preprocess().text


def preprocess_training_data():
    df = import_dataset()
    preprocessed_tweets = []
    print("\nCleaning and parsing tweets from training dataset...\n")
    for i in tqdm(range(0, len(df.index)), desc="Progress", unit="tweets"):
        preprocessed_tweets.append(preprocess(df['text'][i]))
    print()

    preprocessed_df = pd.DataFrame(preprocessed_tweets, columns=['text'])
    preprocessed_df['sentiment'] = df.sentiment
    time = datetime.now().strftime(' %m-%d-%y %H:%M')
    preprocessed_df.to_csv('data/preprocessed_training_data{}'.format(time) + '.data', encoding='utf-8', index=False)


def main():
    preprocess_training_data()


if __name__ == '__main__':
    main()
