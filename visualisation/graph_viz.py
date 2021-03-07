# MOST FREQUENT NEGATIVE TOKENS
# MOST FREQUENT POSITIVE TOKENS
# contains code for most frequent positive/negative tokens
# contains code for average sentiment per day graph
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words

df = pd.read_csv("testing/combined_predicted_election_tweets.csv")
stop_words = set(stopwords.words('english') + ["general", "election"] + list(sklearn_stop_words))

# initialise count vectoriser
cvec = CountVectorizer(stop_words=stop_words)
cvec.fit(df.text)
neg_doc_matrix = cvec.transform(df[df.sentiment == 0].text)
pos_doc_matrix = cvec.transform(df[df.sentiment == 1].text)
neg_tf = np.sum(neg_doc_matrix, axis=0)
pos_tf = np.sum(pos_doc_matrix, axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg, pos], columns=cvec.get_feature_names()).transpose()

term_freq_df.columns = ['negative', 'positive']
# print('n', len(term_freq_df.negative)) # classes balanced so name number of each
# print('p', len(term_freq_df.positive))

term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']


def main():
    # neg_tokens_chart()
    # pos_tokens_chart()
    # sentiment_graph()
    # reformat_date()
    date_freq()


def neg_tokens_chart():
    neg_tokens = term_freq_df.sort_values(by='negative', ascending=False)
    x = neg_tokens.index
    y = neg_tokens['negative'][:50]

    fig = go.Figure(
        data=[go.Bar(x=list(x), y=list(y))],
        layout=go.Layout(
            title=go.layout.Title(text="Top 50 Tokens in Tweets with Predicted Negative Sentiment"),
            xaxis_title="Word Token",
            yaxis_title="Token Frequency",
        )
    )
    fig.update_xaxes(tickangle=45)
    return fig

def pos_tokens_chart():
    pos_tokens = term_freq_df.sort_values(by='positive', ascending=False)
    x = pos_tokens.index

    y = pos_tokens['positive'][:50]

    fig = go.Figure(
        data=[go.Bar(x=list(x), y=list(y))],
        layout=go.Layout(
            title=go.layout.Title(text="Top 50 Tokens in Tweets with Predicted Positive Sentiment"),
            xaxis_title="Word Token",
            yaxis_title="Token Frequency",
        )  
    )
    fig.update_xaxes(tickangle=45)
    return fig


# function reformats date to allow for graph display by day
def reformat_date(df):
    # example = '2019-12-17 23:53:08+00:00' - > 2019-12-17
    # print(example[:10])
    result = df.copy()
    result['date'] = [x[:10] for x in df['date']]
    result.to_csv('data/reformatted_predicted_election_tweets.csv', encoding='utf-8',
                  index=False)


def sentiment_graph():
    # read csv with reformatted date column
    date_df = pd.read_csv("data/reformatted_predicted_election_tweets.csv", parse_dates=['date'],
                          index_col='date')
    aggregate = date_df.resample('D').mean()  # calculates mean sentiment per day
    # print(aggregate)
    fig = go.Figure(
        data=[go.Scatter(x=aggregate.index, y=aggregate.sentiment)],
        layout=go.Layout(
            title=go.layout.Title(text="Average Predicted Sentiment per day"),
            xaxis_title="Date",
            yaxis_title="Average Sentiment",
            xaxis_tickmode='linear'

        )
    )
    return fig


def tweet_freq():
    date_df = pd.read_csv("data/reformatted_predicted_election_tweets.csv", parse_dates=['date'],
                          )

    fig = go.Figure(
        data=[go.Scatter(x=date_df.groupby('date').size().index, y=date_df.groupby('date').size().values)],
        layout=go.Layout(
            title=go.layout.Title(text="Number of Tweets per day"),
            xaxis_title="Date",
            yaxis_title="Frequency of Tweets",
            xaxis_tickmode='linear'

        )
    )

    return fig


if __name__ == '__main__':
    main()
