import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn import preprocessing
import math
import re

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


acquired_dict = {
    'tr':'tr',
    'fa':'fa',
    'dr':'dr',
    "Traded": "tr",
    "Free Agency": "fa",
    "Amateur Draft": "dr",
    "Amateur Free Agent": "fa",
    "Waivers": "tr",
    "Purchased":"tr",
    "Rule 5 Draft": "dr",
    "Expansion Draft": "dr",
    "Conditional Deal": "tr",
    "Amateur Draft--no sign": "dr",
    "MinorLg Draft": "dr",
    "Rune 5 returned": "tr"
}

def inflation_calc(row):
    inf_dict = {
     2017: 1.0,
     2016: 1.021299290023666,
     2015: 1.0341874211554445,
     2014: 1.0354149770208165,
     2013: 1.0522113523096537,
     2012: 1.0676237183898534,
     2011: 1.089717656786951,
     2010: 1.1241149062626115,
     2009: 1.1425534989302544,
     2008: 1.1384885486964882,
     2007: 1.1822013870802828,
     2006: 1.215873015873016,
     2005: 1.2550947260624679,
     2004: 1.297617787188989,
     2003: 1.3324635790389214,
     2002: 1.3626862352679565,
     2001: 1.3843112893206078,
     2000: 1.4234610917537749
    }
    return int(row['salary']*inf_dict[row['year']])

def fixtm(t):
    if t == '2TM' or t == '3TM' or t == '4TM':
        return 'multiple'
    elif t == 'TBD':
        return 'TBR'
    elif t == 'MON':
        return "WSN"
    elif t == 'ANA':
        return 'LAA'
    elif t == 'FLA':
        return 'MIA'
    else: return t

def fix_name(n):
    n1 = (' ').join(n.split('\xa0'))
    n2 = re.sub(r'[^\w\s]','',n1)
    return n2

def train_and_test(cutoff = 1000000):
    train_X,train_y,test_X,test_y = load_and_split_data(cutoff)

    
    lr = LinearRegression()

    lr.fit(train_X, train_y)

    preds = lr.predict(test_X)

    error = np.sqrt(MSE(test_y,preds))
    
    return round(10**error,2)

def train_and_test(cutoff = 1000000):
    train_X,train_y,test_X,test_y = load_and_split_data(cutoff)

    
    lr = LinearRegression()

    lr.fit(train_X, train_y)

    preds = lr.predict(test_X)

    error = np.sqrt(MSE(test_y,preds))
    
    return round(10**error,2)


def cutoff_df(df,cutoff):
    log_10_cut = math.log10(cutoff)
    df = df[df['log10_adj'] >= log_10_cut]
    return df

def test_cutoffs():
    test_cutoffs = [(i+1)*100000 for i in range(20)]
    error_list = []
    for i in test_cutoffs:
        error = train_and_test(i)
        error_list.append(error)
    return test_cutoffs,error_list

def test_elastic_cutoffs():
    test_cutoffs = [(i+1)*100000 for i in range(20)]
    error_list = []
    for i in test_cutoffs:
        error = elastic(i)
        error_list.append(error)
    return test_cutoffs,error_list    


def load_data():
    train = pd.read_pickle('batting_00_16.pkl')
    test = pd.read_pickle('batting_17.pkl')
    return pd.concat([train,test])

def ordered(row):
    if row['name'] == row['np']:
        return row['next_sal']
    else:
        return np.nan

def get_salary_for_next_year():
    df = load_data()
    df = engineer_features(df)
    df = df.sort_values(by = ['name','year'])
    df['next_sal'] = df['log10_adj'].shift(-1)
    df['np'] = df['name'].shift(-1)
    df['next_sal'] = df.apply(ordered,axis=1)
    df = df.dropna()
    df['log10_adj'] = df['next_sal']
    df = df.drop(['next_sal','np'],axis=1)
    
    train = df[df['year']<2016]
    test = df[df['year']==2016]
    
    return train,test

def engineer_features(df):
    df = df[df.pa>200]

    df = df.reset_index()

    df['name'] = df['name'].apply(fix_name)
    #adjust team names
    df['tm'] = df['tm'].apply(fixtm)
    #drop position summary (too many classes), log_sal (unscaled by inflation), rk (same as index)
    df.drop(['pos\xa0summary','log_sal','rk','index'],axis=1,inplace=True)
    #map values in acquired to 3 classes
    df['acquired'] = df['acquired'].map(acquired_dict)
    
    #adjust salary for inflation and take the log-10 for target column
    df['adj_salary'] = df.apply(inflation_calc,axis=1)
    df['log10_adj'] = np.log10(df['adj_salary'])
    
    #get dummy variables for team, hand, and acquired columns
    df = pd.get_dummies(df,columns = ['acquired','bat_hand','tm']).drop(['tm_multiple','bat_hand_rhb','acquired_tr'],axis=1)
    #filter datasets for only batters with more than 200 plate appearances in season

    return df

def scaleColumns(df, cols_to_scale):
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in cols_to_scale:
        df[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])
    return df    

def rescale_numeric(df):
    df = df.reset_index().drop(['index'],axis=1)
    cols = ['g','pa','rbat','rbaser','rdp',
         'rfield',
         'rpos',
         'raa',
         'waa',
         'rrep',
         'rar',
         'war',
         'waawl%',
         '162wl%',
         'owar',
         'dwar',
         'orar',
         'year',
         'ab', 'r', 'h', '2b', '3b', 'hr', 'rbi', 'sb', 'cs', 'bb', 'so', 'ibb',
         'hbp', 'sh', 'sf', 'gidp', 'years_in_mlb']
    df = scaleColumns(df,cols)

    return df

def combine_with_lehman_data(df):
    players = pd.read_csv('baseballdatabank-master/core/People.csv')
    #players = players.set_index('playerID')

    drop_cols = ['deathYear','deathMonth','deathDay','deathCountry','deathState','deathCity',
                'birthYear','birthMonth','birthDay','birthCountry','birthState','birthCity',
                'nameGiven','weight','height','bats','throws','finalGame','retroID','bbrefID']
    players = players.drop(drop_cols,axis=1)
    players['fullname'] = players['nameFirst'] + ' ' + players['nameLast']
    players = players.dropna()

    players['fullname'] = players['fullname'].apply(lambda x: ''.join(re.sub(r'[^\w\s]','',x).split(' ')).lower())
    
    batting = pd.read_csv('baseballdatabank-master/core/Batting.csv')
    bats = batting[batting['yearID'] >= 2000]
    
    bat_join = bats.merge(players,how='left',on='playerID')

    keep_cols = ['yearID',
     'G',
     'AB',
     'R',
     'H',
     '2B',
     '3B',
     'HR',
     'RBI',
     'SB',
     'CS',
     'BB',
     'SO',
     'IBB',
     'HBP',
     'SH',
     'SF',
     'GIDP',
     'debut',
     'fullname']

    bat_join = bat_join[keep_cols]

    bat_join.columns = [x.lower() for x in bat_join.columns]

    bat_join = bat_join.groupby(['fullname','yearid','debut'],axis=0)['g','ab','r','h','2b','3b','hr','rbi','sb','cs','bb','so','ibb','hbp','sh','sf','gidp'].sum().reset_index()
    bat_join['str_g'] = bat_join['g'].apply(str)
    bat_join['str_year'] = bat_join['yearid'].apply(str)
    bat_join['name_g_y'] = bat_join['fullname'] + ' ' + bat_join['str_g'] + ' ' + bat_join['str_year']
    
    df['str_g'] = df['g'].apply(str)
    df['str_year'] = df['year'].apply(str)
    df['name'] = df['name'].apply(fix_aoki_and_castell)
    df['name_g_y'] = df['name'].apply(lambda x: ''.join(x.split(' ')).lower()) + ' ' + df['str_g'] + ' ' + df['str_year']
    
    df = df.merge(bat_join,how='left',on='name_g_y')
    
    df = df.dropna()

    df['debut_year'] = df['debut'].apply(lambda x: int(x.split('-')[0]))

    df['years_in_mlb'] = df['year'] - df['debut_year']
    df['g'] = df['g_x']
    df = df.drop(['g_x','g_y','str_g_x','str_g_y','str_year_x','str_year_y','debut','debut_year','yearid','name_g_y','fullname'],axis=1)
    return df

def fix_aoki_and_castell(name):
    if name == 'Norichika Aoki':
        return 'Nori Aoki'
    elif name == 'Nicholas Castellanos':
        return 'Nick Castellanos'
    else: return name


def load_and_split_data(cutoff = 1):
    #Load dataframes from pickle


    train,test = get_salary_for_next_year()
    
    #Scale inflation and engineer categorical features

    #Combine calculated statistics scraped from baseball-reference with raw stats from Lehman database
    train = combine_with_lehman_data(train)
    test = combine_with_lehman_data(test)
    

    #Rescale numeric features to be (0,1)
    train = rescale_numeric(train)
    test = rescale_numeric(test)
    
    #Cut dataframe by minimum salary
    train = cutoff_df(train,cutoff)
    test = cutoff_df(test,cutoff)
    
    #Split into features and response matrices
    train_y = train['log10_adj']
    test_y = test['log10_adj']
    train_X = train.drop(['name','age','log10_adj','salary','adj_salary'],axis=1)
    test_X = test.drop(['name','age','log10_adj','salary','adj_salary'],axis=1)
    
    return train_X, train_y, test_X, test_y


def plot_with_cutoff(cut,err):
    fig,ax = plt.subplots(figsize=(4,4))

    ax.scatter([i/1000 for i in cut],err)
    ax.set_title("Error Factor vs Cutoff Salary")
    ax.set_ylabel("Error Factor")
    ax.set_xlabel("Minimum Salary (Thousands)")
    ax.set_yticks([1,1.5,2,2.5,3]);
    ax.set_xticks([0,500,1000,1500,2000]);
    plt.tight_layout()

def plot_with_cutoff_and_cols(cut,err,cols):
    fig,ax = plt.subplots(figsize=(4,4))

    ax.scatter([i/1000 for i in cut],err)
    ax.scatter([i/1000 for i in cut],[i/10 for i in cols])
    ax.set_title("Error Factor vs Cutoff Salary")
    ax.set_ylabel("Error Factor")
    ax.set_xlabel("Minimum Salary (Thousands)")
    ax.set_yticks([1,1.5,2,2.5,3]);
    ax.set_xticks([0,500,1000,1500,2000]);
    plt.tight_layout()