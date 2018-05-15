import pickle
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

with open('scaler_object.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('text_bow_object.pkl', 'rb') as f:
    text_bow = pickle.load(f)

with open('link_bow_object.pkl', 'rb') as f:
    link_bow = pickle.load(f)

with open('random_forest_model_full.pkl', 'rb') as f:
    rf = pickle.load(f)



'''authors = pd.read_pickle('author_counts.pkl')


test_input = {'author':'gingermuffinboy','author_flair_text':'','body':'''



'''}

test_df = pd.DataFrame([test_input])

print(test_df)'''


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


def get_author(author):
    if author is not None:
        try: 
            return int(authors.loc[author])
        except:
            return 0
    else:
        return 0

def clean_df(df,authors):
    #remove newline ('\n') characters from text
    df['body'] = df['body'].apply(lambda x: drop_newlines(x))

    #create a binary column that states if the author has a flair
    #possible to do analysis on specific flairs later
    df['has_flair'] = df['author_flair_text'].apply(has_flair)

    df['author_counts'] = df['author'].apply(get_author)

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

    return df


def get_bag_of_words_text(df,text_bow):
    #Convert lemmatized text to "bag of words with top 3000 features"
    text = text_bow.transform(df['body_clean'])
    text_bow_df = pd.DataFrame(text.toarray())

    return text_bow_df


def get_bag_of_words_url(df,link_bow):
    #convert urls to bag of words with 100 features
    links = link_bow.transform(df['links'])
    link_bow_df = pd.DataFrame(links.toarray())

    return link_bow_df

def bow_and_scale(df,text_bow,link_bow):
	
    text = get_bag_of_words_text(df,text_bow)
    urls = get_bag_of_words_url(df,link_bow)
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
    
    full_X_scaled = scaler.transform(full_X)


    return full_X_scaled


'''
test_clean = clean_df(test_df,authors)

test_input = pd.DataFrame(bow_and_scale(test_clean,text_bow,link_bow))

prediction = rf.predict_proba(test_input)

if prediction[0][1] > 0.1:
	print("This post is likely to be downvoted, you may want to reconsider posting it.")
else:
	print("This post is unlikely to be downvoted, post away!")'''
