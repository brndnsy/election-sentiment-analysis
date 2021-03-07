# MOST POPULAR HASHTAGS AS WORDCLOUD
from datetime import datetime
import re
import plotly.express as px
import stylecloud
from nltk import flatten
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words

df = pd.read_csv("testing/combined_predicted_election_tweets.csv")
stop_words = set(stopwords.words('english') + list(sklearn_stop_words))

def gen_wordcloud():
    popular_hashtags = df['hashtags']  # negative sentiment reviews are labelled 0
    # print(df['hashtags'].head())
    hashtag_tokens = flatten([word.split() for word in popular_hashtags if re.search(r"\w", word)]) # flatten to 1 row
    filtered_tokens = list(filter(lambda i: 'election' not in i.lower(), hashtag_tokens)) # remove tags stating the election
    print(len(filtered_tokens))
    filtered_tokens = list(filter(lambda i: 'ge' not in i.lower(), filtered_tokens)) # remove tags stating ge
    print(len(filtered_tokens))
    wordcloud_tokens = " ".join(filtered_tokens)
   
    
    print(len(wordcloud_tokens))
    stylecloud.gen_stylecloud(text=wordcloud_tokens,
                              colors=['#e74c3c', '#3498db', '#000080'],
                              background_color='white',
                              icon_name='fab fa-twitter',
                              size=4096,
                              max_words=50,
                              output_name="assets/hashtags_wordcloud3.png",
                              max_font_size=400,
                              stopwords=stop_words
                              )


def main():
    gen_wordcloud()


if __name__ == '__main__':
    main()
