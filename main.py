import re
from collections import Counter
import dash
import dash_core_components as dcc
import dash_d3cloud
import dash_html_components as html
import pandas as pd  # import pandas python library
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from nltk import flatten
from nltk.tokenize import WordPunctTokenizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier, Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm

tokenizer = WordPunctTokenizer()
plt.style.use('fivethirtyeight')  # 'fivethirtyeight' styling


def import_dataset():
    df = pd.read_csv("data/moviereviewsdataset.csv", encoding='latin-1')  # read csv file
    df = df[df.Sentiment.isnull() == False]  # error check
    df['Sentiment'] = df['Sentiment'].map(int)  # convert string to integer
    df = df[df['SentimentText'].isnull() == False]
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    return df


reviews_dataset = import_dataset()
print("Head of Movie Reviews Dataset: ", reviews_dataset.head())

# remove URL, HTML tags, handle negation words which are split, convert the
# words to lower cases, remove all non-letter characters.

pat_1 = r"(?:\@|https?\://)\S+"
pat_2 = r'#\w+ ?'
combined_pat = r'|'.join((pat_1, pat_2))
www_pat = r'www.[^ ]+'
html_tag = r'<[^>]+>'
negations_ = {"isn't": "is not", "can't": "can not", "couldn't": "could not", "hasn't": "has not",
              "hadn't": "had not", "won't": "will not",
              "wouldn't": "would not", "aren't": "are not",
              "haven't": "have not", "doesn't": "does not", "didn't": "did not",
              "don't": "do not", "shouldn't": "should not", "wasn't": "was not", "weren't": "were not",
              "mightn't": "might not",
              "mustn't": "must not"}
negation_pattern = re.compile(r'\b(' + '|'.join(negations_.keys()) + r')\b')


# handy function to monitor DataFrame creations, then look at our cleaned data.
def data_cleaner(text):
    try:
        stripped = re.sub(combined_pat, '', text)
        stripped = re.sub(www_pat, '', stripped)
        cleantags = re.sub(html_tag, '', stripped)
        lower_case = cleantags.lower()
        neg_handled = negation_pattern.sub(lambda x: negations_[x.group()], lower_case)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        tokens = tokenizer.tokenize(letters_only)
        return (" ".join(tokens)).strip()
    except:
        return 'NC'


def run_classification():
    # Before training, split data into training & validation set.
    # train_test_split splits arrays or matrices into random train and test subsets
    x_train, x_validation, y_train, y_validation = train_test_split(reviews_dataset.SentimentText,
                                                                    reviews_dataset.Sentiment,
                                                                    test_size=.2, random_state=2000)

    # In this part, we will use a feature extraction technique called Tfidf vectorizer of 100,000 features including up
    # to trigram. This technique is a way to convert textual data to numeric form.
    # In the below function, we will use a custom function that reports validation accuracy, the average
    # precision_recall, and the time it took to train and evaluate.

    def acc_summary(pipeline, x_train, y_train, x_test, y_test):
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
        print("accuracy score: {0:.2f}%".format(accuracy * 100))
        print("-" * 80)
        return accuracy, precision, recall, average_precision, fpr, tpr

    names = ["Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection", "Multinomial NB",
             "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron", "Passive-Aggresive", "Nearest Centroid"]
    classifiers = [
        LogisticRegression(),
        LinearSVC(),
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
    vec = TfidfVectorizer()

    def classifier_comparator(vectorizer=vec, n_features=10000, stop_words=None, ngram_range=(1, 1),
                              classifier=zipped_clf):
        result = []
        vectorizer.set_params(stop_words=stop_words, max_features=n_features, ngram_range=ngram_range)
        for n, c in classifier:
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', c)
            ])
            print("Validation result for {}".format(n))
            print(c)
            clf_acc, prec, rec, avg, fp, tp = acc_summary(checker_pipeline, x_train, y_train, x_validation,
                                                          y_validation)
            result.append((n, clf_acc, prec, rec, avg, fp, tp))
        return result

    result = classifier_comparator(n_features=100000, ngram_range=(1, 3))

    saved_clf = 'data/digits_classifier.joblib.pkl'
    _ = joblib.dump(result, saved_clf, compress=9)

    return result


acc = []
names = []
avg_ = []
fp_ = []
tp_ = []

for name, accuracy, prec, rec, avg, fp, tp in joblib.load('data/digits_classifier.joblib.pkl'):
    acc.append(accuracy * 100)
    names.append(name)

    avg_.append(avg * 100)
    fp_.append(fp)
    tp_.append(tp)

trace1 = go.Bar(
    x=names,
    y=acc,
    name='Accuracy percentage (%)'
)
trace3 = go.Bar(
    x=names,
    y=avg_,
    name='Average precision (%)'
)

data = [trace1, trace3]
layout = go.Layout(
    barmode='group')

# handy function to monitor DataFrame creations, then look at our cleaned data.
tqdm.pandas(desc="Progress Bar")


def post_process(data, n=1000000):
    data = data.head(n)
    data['SentimentText'] = data['SentimentText'].progress_map(data_cleaner)
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


processed_reviews_data = post_process(reviews_dataset)

# word cloud for negative sentiment
neg_sent = reviews_dataset[reviews_dataset.Sentiment == 0]  # negative sentiment reviews are labelled 0
neg_review_tokens = flatten([word.split() for word in neg_sent.SentimentText if re.search(r"\w", word)])
neg_wordcloud_tokens = [{"text": a, "value": b} for a, b in Counter(neg_review_tokens).most_common(100)]
# word cloud for positive sentiment
pos_sent = reviews_dataset[reviews_dataset.Sentiment == 1]  # positive sentiment reviews are labelled 1
pos_review_tokens = flatten([word.split() for word in pos_sent.SentimentText if re.search(r"\w", word)])
pos_wordcloud_tokens = [{"text": a, "value": b} for a, b in Counter(pos_review_tokens).most_common(100)]

# instantiate dash application
app = dash.Dash(__name__)
# create html elements

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
    html.Div([
        html.H2("Conducting Sentiment Analysis with Machine Learning"),
        html.Img(src=" assets/mlicon.jpeg")
    ], className="banner"),
    html.Span(
        html.P(
            "The chosen dataset is comprised of 25,000 movie reviews from IMBD, which have been labelled by sentiment "
            "i.e. positive/negative -> 0/1.")
    ),
    html.Div([
        html.Span(html.H3("Word Cloud for Negative Sentiments")),
        dash_d3cloud.WordCloud(
            id='neg_wordcloud',
            words=neg_wordcloud_tokens,
            options={'fontSizes': [15, 150]},
        ),
        html.Span(html.H3("Word Cloud for Positive Sentiments")),
        dash_d3cloud.WordCloud(
            id='pos_wordcloud',
            words=pos_wordcloud_tokens,
            options={'fontSizes': [15, 150]},
        ),
        dcc.Graph(
            id='model-chart',
            figure={
                'data': [trace1, trace3],
                'layout': {
                    'title': 'Machine Learning Models Comparison',
                    # 'plot_bgcolor': colors['background'],
                    # 'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text']
                    }
                }
            }

        )
    ]),

])

if __name__ == '__main__':
    app.run_server(debug=True)
