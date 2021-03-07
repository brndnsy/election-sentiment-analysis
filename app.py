import dash
import dash_core_components as dcc
import dash_html_components as html
import joblib
from plotly import graph_objs as go
from visualisation.graph_viz import neg_tokens_chart, pos_tokens_chart, sentiment_graph, tweet_freq

acc = []
names = []
avg_ = []
fp_ = []
tp_ = []

for name, accuracy, prec, rec, avg, fp, tp in joblib.load(
        'training/data/trained_sentiment_classifier 02-29-20 19:20.pkl'):
    acc.append(accuracy * 100)
    names.append(name)
    
    avg_.append(avg * 100)
    fp_.append(fp)
    tp_.append(tp)

trace1 = go.Bar(
    x=names,
    y=acc, # accuracy
    name='Accuracy percentage (%)'
)
trace3 = go.Bar(
    x=names,
    y=avg_, # average precision score
    name='Average precision (%)'
)

data = [trace1, trace3]
layout = go.Layout(
    barmode='group')

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

metas = [
    {'name': 'description', 'content': 'Sentiment Analysis: UK General Election 2019'},
    {'name': 'title', 'property': 'og:title', 'content': 'Twitter Sentiment Analysis: UK General Election'},
    {'property': 'og:type', 'content': 'Website'},
    {'name': 'image', 'property': 'og:image', 'content': 'https://brndnsy.pythonanywhere.com/static/hashtags_wordcloud3.png'},
    {'name': 'description', 'property': 'og:description', 'content': 'Sentiment Analysis: UK General Election 2019'},
    {'name': 'author', 'content': 'Brandon Charles'}]

app = dash.Dash(__name__, meta_tags=metas)

layout = html.Div([

    html.Div([
        html.Div([html.H2("Twitter Sentiment Analysis - UK General Election"),
                  html.P("Data science project that incorporates natural language processing and machine learning")], className="text-container "
                 ),

        html.Video(src="assets/Earth-Night-Europe-Brexit.mp4", autoPlay=True, loop=True,
                   muted=True, className="video-bg", poster="assets/boris.jpg",
                   title="Sentiment Analysis using Machine Learning")
    ], className="video-container"),

    dcc.Markdown('''
    ## Introduction
    
    After years of speculation and uncertainty regarding the future of the UK's relationship with the EU, a General Election 
    was held on the 12th December 2019 to decide on how exactly _'Brexit'_ would unfold. This project analyses public 
    opinion on the General Election, spanning from the week before it occurred, up until the week afterwards. This was 
    implemented by using machine learning to classify tweets as either positive or negative. 

    ''', className='container',
                 style={'maxWidth': '800px', "white-space": "pre", "overflow-x": "scroll"}),
    
      dcc.Markdown('''
    ***
    ## Data Preprocessing
    
    I utilised a dataset of 1.6million tweets, pre-labelled by sentiment. Prior to training, I had to clean and preprocess 
    each tweet. This involved removing information like urls, punctuation and whitespace, as well as normalising all 
    tweets to lowercase. 
    ''', className='container',
                 style={'maxWidth': '800px', "white-space": "pre", "overflow-x": "scroll"}),
      
    html.Div([html.P('More info on the dataset (from Stanford University) can be found ', 
                     style={'display': 'inline-block', 'white-space': 'pre'}),
              html.A('here.', href='http://help.sentiment140.com/for-students', 
                     target='_blank', style={'display': 'inline-block'})]
             ,className='container', style={'maxWidth': '800px'}),
             
    dcc.Markdown('''
    ***
    ## Training The Model
    I began by splitting the dataset into two sets: the training set (80%) and the test set (20%). I subsequently built a 
    pipeline of 10 machine learning classifiers, which were trained using the training set, and evaluated using the testing 
    set. This is known as supervised learning because the modelling is informed by the data i.e. labelled sentiment. Consequently, 
    I did a comparative analysis of the 10 models in terms of accuracy and precision. As shown by the bar chart 
    below, logistic regression was the highest performing classifer with an accuracy of 80.38%.
    '''.replace('  ', ''), className='container',
                 style={'maxWidth': '800px', "overflow-x": "scroll"}),

    dcc.Graph(
        id='model-chart',
        figure={
            'data': [trace1, trace3],
            'layout': {
                'title': 'Machine Learning Models Comparison',
                # 'plot_bgcolor': colors['background'],
                # 'paper_bgcolor': colors['background'],
                'font': {'color': colors['text']}
                }
        },
    className='container', style={"overflow-x": "scroll", 'maxWidth': '850px'}
                 ),
    
    dcc.Markdown('''
    ***
    ## Webscraping General Election Tweets
    Twitter's existing API didn't allow access to tweets older than a week, so the final dataset was collected via webscraping.
    Ultimately, I collected 90,000 tweets, posted between the 5th and 18th December 2019, that mentioned the General Election.
    I also narrowed down my search criteria to tweets posted from within the UK. After acumulating the data, I repeated 
    the preprocessing steps described previously.
    '''.replace('  ', ''), className='container',
                 style={'maxWidth': '800px', "overflow-x": "scroll"}),

    dcc.Markdown('''
    ***
    ## Classifying General Election Tweets
    For the main stage of the analysis, I employed the pre-trained logistic regression model to classify the sentiment of the
    tweets pertaining to the General Election. Essentially, each of the 90,000 collected tweets were labelled as being either 
    positive or negative (binary classfication).
    
    '''.replace('  ', ''),
                 className='container',
                 style={'maxWidth': '800px', "overflow-x": "scroll"}
                 ),
    
    dcc.Markdown('''
    ***
    ## Visualisations
    I made a wordcloud to convey the prevalence of the most popular hashtags. I also made two bar charts
    to illustrate the frequency and ambivalence of the most popular positive/negative tokens. In addition to this, I made 2 time series graphs to display
    how both the frequency and the average sentiment of the tweets, shifted from the 5th-18th December. Finally, I made this web app to visualise my findings. These
    visualisations can be seen below.
    ***
    '''.replace('  ', ''),
                 className='container',
                 style={'maxWidth': '800px', "overflow-x": "scroll"}
                 ),
    
    html.Div([
        html.Br(),
        html.H5("Hashtag Wordcloud"),
        html.Img(src="./assets/hashtags_wordcloud3.png", style={
            'height': '650px',
            'width': '650px',
            'title' : 'wordcloud hashtag',
            }),
    ],
        className='container',
        style={'text-align': 'center'}

    ),
    dcc.Markdown("***", className='container',
                 style={'maxWidth': '800px', "overflow-x": "scroll"}),
                 
    dcc.Graph(id='pos_tokens', figure=pos_tokens_chart(),className='container',
                 style={'maxWidth': '1000px', "overflow-x": "scroll"}),
    dcc.Markdown('''
    ##### Comments on most popular positive tokens
     * The most popular positive token is "**vote**" which implies that there was an implicit consensus that this was an important election. 
     * "Labour" appears to be the second most popular token. However, there are three different positive tokens that refer to the 
    **Conservative party**: "tory", "tories", "conservatives" and "conservative". Summing their frequencies gives 7724 which makes 
    it the second most positive token, in retrospect. 
    * Similarly summing up the tokens for "Boris" and "Johnson" makes **Boris Johnson** the third most positive token at 5398.
    ***

    '''.replace('  ', ''),
                 className='container',
                 style={'maxWidth': '800px', "overflow-x": "scroll"}
                 ),
    
    dcc.Graph(id='neg_tokens', figure=neg_tokens_chart()
              , className='container', style={"overflow-x": "scroll", 'maxWidth': '1000px'} ),
    
        dcc.Markdown('''
    ##### Comments on most popular negative tokens
    * The most popular negative token is "**labour**".
    * Summing up the tokens for "Boris" and "Johnson" makes **Boris Johnson** the third most negative token at 3550.
    * Combining "tory" and "tories" would make the **Conservative party** the fourth most negative token at 3228.
    ***
    '''.replace('  ', ''),
                 className='container',
                 style={'maxWidth': '800px', "overflow-x": "scroll"}
                 ),
    
    dcc.Graph(id='sent_date', figure=sentiment_graph(),
    className='container', style={'maxWidth': '1000px', "overflow-x": "scroll"}),
    
    dcc.Markdown('''
    ##### Comments on sentiment over time
    * The average sentiment for every day was relatively positive given values closer to 1 than 0.
    * Despite this, there is a slight negative correlation over time. 
    * There was a steep increase in sentiment the day before the election, and a sharp decline on the election day itself.
    ***
    '''.replace('  ', ''),
                 className='container',
                 style={'maxWidth': '800px', "overflow-x": "scroll"}
                 ),
    dcc.Graph(id='tweet_freq', figure=tweet_freq(),
    className='container', style={'maxWidth': '1000px', "overflow-x": "scroll"}),
    
    dcc.Markdown('''
    ##### Comments on tweet frequency over time
    * There was a steady increase up until the 11th December, then there was a 350% increase from the 11th-12th December.
    * There was a 20% decrease the day after election day, then the aforementioned pattern repeats itself in a reverse manner.
    ***
    '''.replace('  ', ''),
                 className='container',
                 style={'maxWidth': '800px', "overflow-x": "scroll"}
                 ),
])

app.layout = layout

if __name__ == '__main__':
    # app.run_server(debug=True, threaded=True)
    app.run_server(debug=True)
    app.config.requests_pathname_prefix = app.config.routes_pathname_prefix.split('/')[-1]