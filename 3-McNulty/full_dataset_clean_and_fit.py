'''
Script to  preform all data cleaning needed for classification modeling of reddit comments.
'''

import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word
from urllib.parse import urlsplit


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split


nltk.download('stopwords')
nltk.download('wordnet')

def is_downvoted(score):
    if score < 0:
        return 1
    else: return 0

def avg_word(sentence):
    words = sentence.split()
    if len(words) > 0:
        return sum(len(word) for word in words if len(word)<20)/len(words)
    else:
        return 0


def extract_md_link(post):
    doc = etree.fromstring(markdown.markdown(post))
    links = []
    for link in doc.xpath('//a'):
        links.append(link.get('href'))    
    return ' '.join(links)

def drop_newlines(post):
    return ' '.join(post.split('\n'))

def get_link(post):

    myString_list = [item for item in post.split(" ")]
    url_list = []
    for item in myString_list:
        try:
            a = re.search("(?P<url>https?://[^\s]+)", item) or re.search("(?P<url>www[^\s]+)", item)
            a = a.group('url')
            b = '.'.join(urlsplit(a).hostname.split('.')[-2:])
            
            url_list.append(b)
        except:
            pass

    return ' '.join(url_list)


def replace_md_links(body):
    ans = re.findall(r"\[(.*?)\]\((.*?)\)",body)
    text = ' '.join([a[0] for a in ans])
    ret = re.sub(r"\[(.*?)\]\((.*?)\)",'',body)
    new_body = ret + ' ' + text
    return ' '.join([x for x in new_body.split(' ') if len(x) < 20])
    
def has_flair(flair):
    if type(flair) == float:
        return 0
    else: return 1

def filter_deleted(author):
    if author == '[deleted]': return 1
    else: return 0

def remove_deleted(author):
    if author == '[deleted]': return None
    else: return author



def clean_df(df,filename='cleaned_data'):
    #reset index and remove irrelavent columns

    df = df.reset_index()

    drop_cols = [
    'index',
    'score_hidden',
    'name','downs',
    'subreddit_id',
    'subreddit',
    'controversiality',
    'gilded',
    'ups',
    'distinguished',
    'removal_reason',
    'archived',
    'author_flair_css_class'
    ]

    df = df.drop(drop_cols,axis=1)

    #remove nulls in body and retrieved_on columns (about 600 rows removed)
    df = df.dropna(subset=['body','retrieved_on'],axis=0)
    df = df.reset_index()
    df = df.drop('index',axis=1)

    #convert score into a binary 'downvoted' column (target)
    df['downvoted'] = df['score'].apply(lambda x: is_downvoted(x))

    #remove newline ('\n') characters from text
    df['body'] = df['body'].apply(lambda x: drop_newlines(x))

    #convert timestamps to pandas datetime object and calculate age of post 
    #when it was added to the database
    df['created_utc'] = pd.to_datetime(df['created_utc'],unit='s')
    df['retrieved_on '] = pd.to_datetime(df['retrieved_on'],unit='s')
    df['age_retrieved'] = df['retrieved_on '] - df['created_utc']

    #remove '[deleted]' authors, create binary 'author_deleted column'
    df['author_deleted'] = df['author'].apply(filter_deleted)
    df['author_fix'] = df['author'].apply(remove_deleted)

    #create a column that counts how many times the author has posted in the 
    #subreddit
    df['author_counts'] = df.groupby(['author_fix'])['body'].transform('count').fillna(0)

    #create a binary column that states if the author has a flair
    #possible to do analysis on specific flairs later
    df['has_flair'] = df['author_flair_text'].apply(has_flair)

    #get wordcount of each post
    df['word_count'] = df['body'].apply(lambda x: len(str(x).split(' ')))

    #get average word length (words shorter than 20 chars to exclude links)
    df['avg_word'] = df['body'].apply(lambda x: avg_word(x))

    #extract urls from posts and return the domains in the post
    df['links'] = df['body'].apply(lambda x: get_link(x))

    #load in list of stopwords and get a count of stopwords in each post
    stop = stopwords.words('english')
    df['stopwords'] = df['body'].apply(lambda x: len([x for x in x.split() if x in stop]))

    #get digits in post
    df['numerics'] = df['body'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

    #get count of words in ALL CAPS (not including one letter words)
    df['upper'] = df['body'].apply(lambda x: len([x for x in x.split() if x.isupper() and len(x)>1]))

    #clean text: convert to lowercase, remove links, remove stopwords
    df['body_clean'] = df['body'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df['body_clean'] = df['body_clean'].apply(replace_md_links)
    df['body_clean'] = df['body_clean'].str.replace('[^\w\s]','')
    df['body_clean'] = df['body_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    #lemmatize the words (remove endings like ly, ing, etc)
    df['body_clean']=df['body_clean'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    #remove most common words and any word that only occurs once in the post
    #may need to increase number of words removed once the dataset gets bigger
    freq = pd.Series(' '.join(df['body_clean']).split()).value_counts()[:20]
    freq = list(freq.index)
    df['body_clean'] = df['body_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    lowfreq = pd.Series(' '.join(df['body_clean']).split()).value_counts()
    only_one = lowfreq[lowfreq == 1]
    df['body_clean']=df['body_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in only_one))

    #I would like to use the TextBlob correct() method to spell correct the text, but it 
    #seems like it will take way too long

    #do sentiment analysis on the cleaned text
    df['sentiment'] = df['body_clean'].apply(lambda x: TextBlob(x).sentiment[0])

    df.to_pickle('./data/{0}.pkl'.format(filename))


def get_bag_of_words_text(df):
    #Convert lemmatized text to "bag of words with top 3000 features"
    bow = CountVectorizer(max_features=3000, lowercase=True, ngram_range=(1,3),analyzer = "word")
    text_bow = bow.fit_transform(df['body_clean'])
    text_bow_df = pd.DataFrame(text_bow.toarray())

    return text_bow_df


def get_bag_of_words_url(df):
    #convert urls to bag of words with 100 features
    bow = CountVectorizer(max_features=100, lowercase=True, ngram_range=(1,1),analyzer = "word")
    link_bow = bow.fit_transform(df['links'])
    link_bow_df = pd.DataFrame(link_bow.toarray())

    return link_bow_df

def get_x_y_dfs(df):

    y = df['downvoted']
    text = get_bag_of_words_text(df)
    urls = get_bag_of_words_url(df)
    good_cols = df[[
    'word_count',
    'avg_word',
    'stopwords',
    'numerics',
    'upper',
    'sentiment',
    'has_flair',
    'author_counts'
    ]]

    full_X = good_cols.merge(text, how='outer', left_index=True, right_index=True).merge(urls,how='outer', left_index=True, right_index=True)

    return full_X, y

def split_with_bow():
    cleaned = pd.read_pickle('./data/cleaned_data.pkl')
    X,y = get_x_y_dfs(cleaned)

    return X,y

def rescale_dataset(X,y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    #X_test_scaled = scaler.transform(X_test)

    return X_scaled,y

X,y = split_with_bow()

Xs,y = rescale_dataset(X,y)


rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=50, 
    max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
    min_samples_leaf=2, min_samples_split=10, min_weight_fraction_leaf=0.0, 
    n_estimators=200, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)

rf.fit(Xs,y)

with open('random_forest_model_full.pkl', 'wb') as handle:
    pickle.dump(rf, handle, protocol=pickle.HIGHEST_PROTOCOL)




