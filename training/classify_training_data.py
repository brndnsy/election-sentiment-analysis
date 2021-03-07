from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier, Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, roc_curve, \
    confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm
from datetime import datetime
import joblib
import pandas as pd
import tkinter as tk
from tkinter import filedialog


# file dialog to import preprocessed training data data file
def load_csv():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    preprocessed_df = pd.read_csv(file_path)
    root.update()
    root.destroy()

    print(preprocessed_df[preprocessed_df.isnull().any(axis=1)].head())
    # print(preprocessed_df[preprocessed_df.isnull().any(axis=1)])
    print("Null b4:", len(preprocessed_df[preprocessed_df.isnull().any(axis=1)]))
    # drop null rows & update dataframe
    preprocessed_df.dropna(inplace=True)
    preprocessed_df.reset_index(drop=True, inplace=True)
    print("Null after:", len(preprocessed_df[preprocessed_df.isnull().any(axis=1)]))

    return preprocessed_df


def main():
    df = load_csv()
    SEED = 2000
    # cross validation technique - train_test_split
    x_train, x_validation, y_train, y_validation = train_test_split(df.text, df.sentiment, test_size=.2,
                                                                    random_state=SEED)
    print("\nTrain set has {0} total entries: {1:.2f}% negative, {2:.2f}% positive".format(len(x_train), (
            len(x_train[y_train == 0]) / (len(x_train) * 1.)) * 100, (len(x_train[y_train == 1]) / (
            len(x_train) * 1.)) * 100))
    print("Validation set has {0} total entries: {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation), (
            len(x_validation[y_validation == 0]) / (len(x_validation) * 1.)) * 100, (len(
        x_validation[y_validation == 1]) / (len(x_validation) * 1.)) * 100))
    if len(x_validation[y_validation == 0]) / (len(x_validation) * 1.) > 0.5:
        null_accuracy = len(x_validation[y_validation == 0]) / (len(x_validation) * 1.)
    else:
        null_accuracy = 1. - (len(x_validation[y_validation == 0]) / (len(x_validation) * 1.))

    print("Null accuracy: {0:.2f}%".format(null_accuracy * 100))

    # feature extraction technique - Tfidf vectorizer of 100,000 features up to trigram.
    # this works to vectorise the text
    # validation accuracy & precision_recall given for each classifier
    # built custom pipeline to evaluate different models

    def acc_summary(pipeline, x_test, y_test):

        sentiment_fit = pipeline.fit(x_train, y_train)
        y_pred = sentiment_fit.predict(x_test)

        # Compute the accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Compute the precision and recall
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        # Compute the average precision
        average_precision = average_precision_score(y_test, y_pred)

        # conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0, 1]))
        # confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
        #                          columns=['predicted_negative', 'predicted_positive'])
        #
        # print("Confusion Matrix\n")
        # print(confusion)

        # print("Classification Report\n")
        print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))


        fpr, tpr, _ = roc_curve(y_test, y_pred)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        print("Accuracy score: {0:.2f}%".format(accuracy * 100))
        if accuracy > null_accuracy:
            print("\nmodel is {0:.2f}% more accurate than null accuracy".format((accuracy - null_accuracy) * 100))
        return accuracy, precision, recall, average_precision, fpr, tpr

    names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection", "Multinomial NB",
             "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron", "Passive-Aggresive", "Nearest Centroid"]
    classifiers = [
        LogisticRegression(solver='lbfgs', max_iter=3000),
        LinearSVC(),
        # LinearSVC with L1-based feature selection
        Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
            ('classification', LinearSVC(penalty="l2"))]),
        MultinomialNB(),
        BernoulliNB(),
        RidgeClassifier(),
        AdaBoostClassifier(),
        Perceptron(),
        PassiveAggressiveClassifier(),
        NearestCentroid()
    ]
    zipped_clf = zip(names, classifiers)  # grouped classification methods

    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

    def compare_classifier_methods(vectorizer=TfidfVectorizer(), n_features=10000, ngram_range=(1, 3),
                                   classifier=zipped_clf):
        result = []
        vectorizer.set_params(max_features=n_features, ngram_range=ngram_range)
        for n, c in tqdm(classifier, total=len(classifiers), desc="\nProgress", unit="classifiers"):
            comparison_pipeline = Pipeline([
                ('vectoriser', vectorizer),
                ('classifier', c)
            ])
            print('\n', '-' * 100)
            print("Validation result for {}".format(n), ':')
            # print(c)
            clf_acc, prec, rec, avg, fp, tp = acc_summary(comparison_pipeline, x_validation,
                                                          y_validation)
            result.append((n, clf_acc, prec, rec, avg, fp, tp))

        return result

    result = compare_classifier_methods()

    # save classifier model along with build date
    time = datetime.now().strftime(' %m-%d-%y %H:%M')
    clf_path = './data/trained_sentiment_classifier{}'.format(time) + '.pkl'
    with open(clf_path, 'wb') as f:  # write in binary mode to pickle file
        joblib.dump(result, f)

    return result


if __name__ == '__main__':
    main()
