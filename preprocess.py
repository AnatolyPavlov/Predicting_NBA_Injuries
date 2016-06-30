from datetime import datetime
from collections import defaultdict
import pandas as pd
import os
import json
import numpy as np
import re
import cPickle as pickle


# function to extract information from the json files
def json_extract(file):
    with open('data/{}'.format(file)) as f:
        data = json.load(f)
        cols = data['resultSets'][0]['headers']
        vals = data['resultSets'][0]['rowSet']
    return cols, vals


# function to create dataframe from the json information
# keyword = gamelog, season_stats, or heights_weights
def create_df(keyword, add_year=False):

    fns = os.listdir('data/')

    cols = json_extract('2013_{}.json'.format(keyword))[0]
    if add_year:
        cols += ['YEAR']
    df = pd.DataFrame(columns=cols)

    for fn in fns:
        if keyword in fn:
            tmp_cols, tmp_vals = json_extract(fn)
            df_tmp = pd.DataFrame(tmp_vals, columns=tmp_cols)
            if add_year:
                df_tmp['YEAR'] = int(fn[0:4])
            df = df.append(df_tmp)
            del df_tmp, tmp_cols, tmp_vals
    return df


def parse_date(df, date_col, create_sep_cols=True):

    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    if create_sep_cols:
        date = df[date_col]
        df['YEAR'] = date.apply(lambda x: x.year)
        df['MONTH'] = date.apply(lambda x: x.month)
        df['DAY'] = date.apply(lambda x: x.day)

    return df


def clean_notes(x):
    found = re.findall(r'\s\(\w\)', x)
    if found:
        return x.replace(found[0], '')
    else:
        return x


# preprocess injury df
def prep_injury(df, gamelog_df):

    drop_vars = ['Unnamed: 0', 'Team']
    df.drop(drop_vars, axis=1, inplace=True)

    # converting the Date column to datetime objects
    df = parse_date(df, 'Date')
#     df['Date'] = pd.to_datetime(df['Date'])

    # filter out all events not directly related to basketball
    df = df[(~df['Notes'].str.contains('flu')) &
            (~df['Notes'].str.contains('rest')) &
            (~df['Notes'].str.contains('jail')) &
            (~df['Notes'].str.contains('ill')) &
            (~df['Notes'].str.contains('asthma')) &
            (~df['Notes'].str.contains('virus')) &
            (~df['Notes'].str.contains('return')) &
            (~df['Notes'].str.contains('pneumonia')) &
            (~df['Notes'].str.contains('coach')) &
            (~df['Notes'].str.contains('sister')) &
            (~df['Notes'].str.contains('Fined')) &
            (~df['Notes'].str.contains('flu')) &
            (~df['Notes'].str.contains('GM')) &
            (~df['Notes'].str.contains('flu')) &
            (~df['Notes'].str.contains('team')) &
            (~df['Notes'].str.contains('canal')) &
            (~df['Notes'].str.contains('food')) &
            (~df['Notes'].str.contains('virus')) &
            (~df['Notes'].str.contains('wife')) &
            (~df['Notes'].str.contains('asthma')) &
            (~df['Notes'].str.contains('chin')) &
            (~df['Notes'].str.contains('headache')) &
            (~df['Notes'].str.contains('anemia')) &
            (~df['Notes'].str.contains('dizziness')) &
            (~df['Notes'].str.contains('cold')) &
            (~df['Notes'].str.contains('throat')) &
            (~df['Notes'].str.contains('molar')) &
            (~df['Notes'].str.contains('dizziness')) &
            (~df['Notes'].str.contains('rash')) &
            (~df['Notes'].str.contains('stomach ache')) &
            (~df['Notes'].str.contains('bronchitis')) &
            (~df['Notes'].str.contains('concussion')) &
            (~df['Notes'].str.contains('recover')) &
            (~df['Notes'].str.contains('mump'))]

    # clean notes
    df['Notes'] = df['Notes'].apply(clean_notes)

    # stripping blank spaces from player names
    df['Player'] = df['Player'].apply(lambda x: x.strip())

    # removing periods from names like C.J.
    df['Player'] = df['Player'].apply(lambda x: ''.join(x.split('.'))
                                      if re.match(r'\w\.\w\.', x) else x)

    # removing characters like (a) and (b)
    df['Player'] = df['Player'].apply(lambda x: ' '.join(x.split()[:2])
                                      if re.match(r'.+\(.+\)', x) else x)

    df[df['Player'] == '(William) Tony Parker']['Player'] = 'Tony Parker'
    df = df[df['Player'] != '']

    unique_players = gamelog_df['PLAYER_NAME'].unique()
    for player in unique_players:
        df['Player'] = df['Player'].apply(lambda x: player
                                          if player in x else x)
    return df


# preprocess
def prep_gamelog(df):

    # converting the Date column to datetime objects
    df = parse_date(df, 'GAME_DATE')

    drop_vars = ['SEASON_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
                 'MATCHUP', 'WL', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
                 'VIDEO_AVAILABLE']
    df.drop(drop_vars, axis=1, inplace=True)

    return df


def start_end_season(keyword, add_year=False):

    fns = os.listdir('data/')
    cols = json_extract('2013_{}.json'.format(keyword))[0]
    if add_year:
        cols += ['YEAR']
    df = pd.DataFrame(columns=cols)

    season_range = []
    for fn in fns:
        if keyword in fn:
            cols, vals = json_extract(fn)
            df = pd.DataFrame(vals, columns=cols)
            season_range.append((datetime.strptime(df['GAME_DATE'].min(),
                                                   '%Y-%m-%d'),
                                 datetime.strptime(df['GAME_DATE'].max(),
                                                   '%Y-%m-%d')))
    return season_range


def create_feat_mat(df, data_window=14):
    feat_mat = pd.DataFrame()
    player_id = df['PLAYER_ID'].unique().tolist()
    for id in player_id:
        tmp = df[df['PLAYER_ID'] == id]
        roll_mean_df = pd.rolling_mean(tmp, data_window)
        feat_mat = feat_mat.append(roll_mean_df)
    return feat_mat


def pickles_to_pandas(keyword, add_year=False):

    fns = os.listdir('data/')
    with open('data/2014_{}_stats.pickle'.format(keyword), 'r') as f:
        pkl_f = pickle.load(f)
        cols = pkl_f[0]['resultSets'][0]['headers']
    df = pd.DataFrame(columns=cols)
    for fn in fns:
        if keyword in fn:
            with open('data/{}'.format(fn), 'r') as f:
                pkl_f = pickle.load(f)
            for item in pkl_f:
                tmp_cols = item['resultSets'][0]['headers']
                tmp_vals = item['resultSets'][0]['rowSet']
                df_tmp = pd.DataFrame(tmp_vals, columns=tmp_cols)
                if add_year:
                    df_tmp['YEAR'] = int(fn[0:4])
                df = df.append(df_tmp)
                del df_tmp, tmp_cols, tmp_vals
    return df


# shuffles the rows of a dataframe
def shuffle_rows(df):
    return df.reindex(np.random.permutation(df.index))


# function that aggregates the stats within a window of specified days
def agg_stats(df, window=14):
    columns = df.columns.tolist()
    cat_dict = defaultdict(list)
    for i in xrange(0, len(df)):
        for col in columns:
            #create a dictionary to store the stats for each category
            cat_dict[col].append(np.nanmean(df[col].iloc[i-window: i]))
        # create a new column that records the number of games played within the window
        cat_dict['GAMES_PLAYED_IN_WINDOW'].append(np.nansum(df['GAMES_PLAYED'].iloc[i-window: i]))
        cat_dict['B2B_PLAYED_IN_WINDOW'].append(np.nansum(df["BACK_TO_BACKS"].iloc[i-window: i]))
    return pd.DataFrame(cat_dict)


# defines the start of the season for each row
def define_season(x):
    for season in season_dt_range:
        if (x >= season[0]) & (x <= season[1]):
            return season[0]


# custom apply function to match stats from other dataframes to the
# feature matrix
# appends the height, weight, and age according to the player and the year
def add_bi_comp_feat(df, player, start_season, feat):
    if df[(df['PLAYER_NAME'] == player) &
              (df['YEAR'] == start_season.year)][feat].empty:
        return None
    return df[(df['PLAYER_NAME'] == player) &
              (df['YEAR'] == start_season.year)][feat].values[0]