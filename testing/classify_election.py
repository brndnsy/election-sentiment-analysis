# Brandon Charles 1505211
# uses logistic regression model for predictive visualisation

from datetime import datetime
from plotly.express import pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os, glob
from tqdm import tqdm


def load_csv(file):
    preprocessed_df = pd.read_csv("../webscraping/output/" + file)
    print(preprocessed_df[preprocessed_df.isnull().any(axis=1)].head())
    print("Null count:", len(preprocessed_df[preprocessed_df.isnull().any(axis=1)]))
    if len(preprocessed_df[preprocessed_df.isnull().any(axis=1)]) > 0:
        print("*Null values removed*")
    # drop null rows & update dataframe
    preprocessed_df.dropna(inplace=True)
    preprocessed_df.reset_index(drop=True, inplace=True)
    print("Final dataframe length: ", len(preprocessed_df), "with null count:",
          len(preprocessed_df[preprocessed_df.isnull().any(axis=1)]))
    print("Null count:", len(preprocessed_df[preprocessed_df.isnull().any(axis=1)]))

    return preprocessed_df


def main():
    training_data = pd.read_csv('../data/preprocessed_training_data 01-26-20 04:07.csv')
    training_data.dropna(inplace=True)
    training_data.reset_index(drop=True, inplace=True)
    df = training_data
    SEED = 2000

    # cross validation technique - train_test_split
    x_train, x_validation, y_train, y_validation = train_test_split(df.text, df.sentiment, test_size=.2,
                                                                    random_state=SEED)
    print("\nThe Training data is a corpus comprised of {0} total entries: {1:.2f}% negative, {2:.2f}% positive".format(
        len(x_train), (
                              len(x_train[y_train == 0]) / (len(x_train) * 1.)) * 100, (len(x_train[y_train == 1]) / (
                len(x_train) * 1.)) * 100))
    print("The validation set has {0} total entries: {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation), (
            len(x_validation[y_validation == 0]) / (len(x_validation) * 1.)) * 100, (len(
        x_validation[y_validation == 1]) / (len(x_validation) * 1.)) * 100))
    if len(x_validation[y_validation == 0]) / (len(x_validation) * 1.) > 0.5:
        null_accuracy = len(x_validation[y_validation == 0]) / (len(x_validation) * 1.)
    else:
        null_accuracy = 1. - (len(x_validation[y_validation == 0]) / (len(x_validation) * 1.))

    print("Null accuracy: {0:.2f}%".format(null_accuracy * 100))

    def acc_summary(x_test, y_test):

        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)
        # Compute the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Compute the precision and recall
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        # Compute the average precision
        average_precision = average_precision_score(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_pred)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        print("Accuracy score: {0:.2f}%".format(accuracy * 100),
              "({0:.2f}% more accurate than null accuracy)".format((accuracy - null_accuracy) * 100))

        return accuracy, precision, recall, average_precision, fpr, tpr, sentiment_fit

    # build pipeline - list of tuples as parameters
    vec = TfidfVectorizer().set_params(max_features=10000, ngram_range=(1, 3))
    pipeline = Pipeline([
        ('vectoriser', vec),
        ('classifier', LogisticRegression(solver='lbfgs', max_iter=4000))
    ])
    # acc_summary(x_validation, y_validation)
    predictor_model = acc_summary(x_validation, y_validation)[-1]
    path = "../webscraping/output/"
    # list of all scraped twitter .csv files for each of the 14 days
    scraped_files = [i for i in os.listdir(path) if i.endswith('.csv')]
    for f in tqdm(scraped_files):
        tweets = load_csv(f)
        file_name = os.path.splitext(f)[0] + ' predicted.csv'
        # main(get_election_tweets, file_name)
        predict_sentiment(tweets, file_name, predictor_model)

    # clf_acc, prec, rec, avg, fp, tp, predictor_model = acc_summary(x_validation, y_validation)
    # clf_acc, prec, rec, avg, fp, tp = acc_summary(x_validation, y_validation)

    # print(acc_summary(x_validation, y_validation)[-1])


def predict_sentiment(get_election_tweets, file_name, predictor_model):
    predictions = []
    for i in [get_election_tweets['text']]:
        predictions.extend(predictor_model.predict(i))
    #     build new csv file including updated sentiment column with predictions
    predicted_election_tweets = pd.DataFrame({
        'text': get_election_tweets['text'],
        'sentiment': predictions,
        'date': get_election_tweets['date'],
        'hashtags': get_election_tweets['hashtags']})
    predicted_election_tweets.to_csv('predicted/{}'.format(file_name), encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
    # ranged()
