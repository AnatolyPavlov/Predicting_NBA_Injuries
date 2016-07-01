import pandas as pd
import os
import json
from datetime import datetime
import numpy as np
import re
from collections import defaultdict
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
def prep_injury(df):

    # dropping unneeded columns
    df.drop(['Unnamed: 0', 'Team'], axis=1, inplace=True)

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

    # manually correcting Tony Parker's name
    df[df['Player'] == '(William) Tony Parker']['Player'] = 'Tony Parker'

    # removing all injuries without a name
    df = df[df['Player'] != '']

    '''
    **************************************
    uses a variable (gamelog) outside of the function
    **************************************
    '''
    unique_players = gamelog_df['PLAYER_NAME'].unique()
    for player in unique_players:
        df['Player'] = df['Player'].apply(lambda x: player
                                          if player in x else x)
    return df


# preprocess the gamelog
def prep_gamelog(df):

    # converting the Date column to datetime objects
    df = parse_date(df, 'GAME_DATE')

    drop_vars = ['SEASON_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
                 'WL', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
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
            season_range.append((datetime.strptime(df['GAME_DATE'].min(), '%Y-%m-%d'),
                                 datetime.strptime(df['GAME_DATE'].max(), '%Y-%m-%d')))
    return season_range


def create_feat_mat(df, data_window=21):
    feat_mat = pd.DataFrame()
    player_id = df['PLAYER_ID'].unique().tolist()
    for id in player_id:
        tmp = df[df['PLAYER_ID'] == id]
        roll_mean_df = pd.rolling_mean(tmp, data_window)
        feat_mat = feat_mat.append(roll_mean_df)
    return feat_mat


def pace_tracking(keep_cols, keyword):
    df = pickles_to_pandas(keyword, add_year=True)
    df.drop(df.columns - keep_cols, axis=1, inplace=True)
    df.rename(columns={'GAME_ID': 'GAME_ID_{}'.format(keyword.upper()),
                     'PLAYER_NAME': 'PLAYER_NAME_{}'.format(keyword.upper()),
                     'PLAYER_ID': 'PLAYER_ID_{}'.format(keyword.upper()),
                     'YEAR': 'YEAR_{}'.format(keyword.upper())},
              inplace=True)
    return df


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


# determines where the game took place depending on the 'vs' and '@'
# in the column
def game_loc(x):
    if 'vs.' in x:
        return city_abbrv[re.split(r' vs. ', x)[0]]
    elif '@' in x:
        return city_abbrv[re.split(r' @ ', x)[1]]


# function that aggregates the stats within a window of specified days
def agg_stats(df, window=14):
    columns = df.columns.tolist()
    cat_dict = defaultdict(list)
    for i in xrange(0, len(df)):
        for col in columns:

            #create a dictionary to store the stats for each category
            cat_dict[col].append(np.nanmean(df[col].iloc[i-window: i]))

        # create new columns that records the number of games played,
        # the number of back-to-backs, and the total miles traveled
        # within the window
        cat_dict['GAMES_PLAYED_IN_WINDOW'].append(np.nansum(df['GAMES_PLAYED'].iloc[i-window: i]))
        cat_dict['B2B_PLAYED_IN_WINDOW'].append(np.nansum(df["BACK_TO_BACKS"].iloc[i-window: i]))
        cat_dict['TOTAL_MILES_TRAVELED'].append(np.nansum(df["MILES_TRAVELED"].iloc[i-window: i]))
    return pd.DataFrame(cat_dict)


# defines the start of the season for each row
def define_season(x):
    for season in season_dt_range:
        if (x >= season[0]) & (x <= season[1]):
            return season[0]


ss_df = create_df('season_stats', add_year=True)
hw_df = create_df('heights_weights', add_year=True)
# convert the weights in the height/weight dataframe to floats
hw_df['PLAYER_WEIGHT'] = hw_df['PLAYER_WEIGHT'].apply(lambda x: float(x))

gamelog_df = create_df('gamelog')
gamelog_df = prep_gamelog(gamelog_df)
injury_df = pd.read_csv('data/injuries.csv')
injury_df = prep_injury(injury_df)

gamelog = gamelog_df
injury = injury_df

start_data = datetime.strptime('10-29-2013', '%m-%d-%Y')
gamelog = gamelog[gamelog['GAME_DATE'] >= start_data]
injury = injury[injury['Date'] >= start_data]
hw_df = hw_df[hw_df['YEAR'] >= 2013]
injury.rename(columns={'Player': 'Player_Injury', 'Date': 'Date_Injury'},
              inplace=True)

pace_keep_cols = ['GAME_ID', 'PACE', 'PLAYER_ID', 'PLAYER_NAME', 'YEAR']
tracking_keep_cols = ['GAME_ID', 'SPD', 'DIST', 'PLAYER_ID', 'PLAYER_NAME',
                      'YEAR']

tracking = pace_tracking('tracking')
pace = pace_tracking('pace')

#pace = pickles_to_pandas('pace', add_year=True)
#pace_keep_cols = ['GAME_ID', 'PACE', 'PLAYER_ID', 'PLAYER_NAME', 'YEAR']
#pace_drop_cols = pace.columns - pace_keep_cols
#pace.drop(pace_drop_cols, axis=1, inplace=True)
#pace.rename(columns={'GAME_ID': 'GAME_ID_PACE',
#                     'PLAYER_NAME': 'PLAYER_NAME_PACE',
#                     'PLAYER_ID': 'PLAYER_ID_PACE',
#                     'YEAR': 'YEAR_PACE'},
#            inplace=True)
#
#tracking = pickles_to_pandas('tracking', add_year=True)
#tracking_keep_cols = ['GAME_ID', 'SPD', 'DIST', 'PLAYER_ID', 'PLAYER_NAME',
#                      'YEAR']
#tracking_drop_cols = tracking.columns - tracking_keep_cols
#tracking.drop(tracking_drop_cols, axis=1, inplace=True)
#tracking.rename(columns={'GAME_ID': 'GAME_ID_TRACKING',
#                     'PLAYER_NAME': 'PLAYER_NAME_TRACKING',
#                     'PLAYER_ID': 'PLAYER_ID_TRACKING',
#                     'YEAR': 'YEAR_TRACKING'},
#            inplace=True)


# merging the gamelogs with pace and tracking statistics
gamelog = gamelog.merge(pace, left_on=['GAME_ID', 'PLAYER_ID'],
                             right_on=['GAME_ID_PACE', 'PLAYER_ID_PACE'])
gamelog = gamelog.merge(tracking, left_on=['GAME_ID', 'PLAYER_ID'],
                                right_on=['GAME_ID_TRACKING', 'PLAYER_ID_TRACKING'])

city_abbrv = {'ATL': 'Atlanta',
              'BOS': 'Boston',
              'CLE': 'Cleveland',
              'GSW': 'Oakland',
              'OKC': 'Oklahoma+City',
              'SAS': 'San+Antonio',
              'POR': 'Portland',
              'MIA': 'Miami',
              'IND': 'Indiana',
              'TOR': 'Toronoto',
              'CHI': 'Chicago',
              'BKN': 'Brooklyn',
              'CHA': 'Charlotte',
              'NYK': 'New+York',
              'DET': 'Detroit',
              'PHI': 'Philadelphia',
              'ORL': 'Orlando',
              'MIL': 'Milwaukee',
              'WAS': 'Washington+DC',
              'DEN': 'Denver',
              'DAL': 'Dallas',
              'MIN': 'Minnesota',
              'LAC': 'Los+Angeles',
              'HOU': 'Houston',
              'LAL': 'Los+Angeles',
              'MEM': 'Memphis',
              'PHX': 'Phoenix',
              'NOP': 'New+Orleans',
              'UTA': 'Utah',
              'SAC': 'Sacramento'}

gamelog['GAME_LOCATION'] = gamelog['MATCHUP'].apply(lambda x: game_loc(x))

# creating dataframe of everyday in a season
season_dt_range = start_end_season('gamelog')

seasons = pd.DataFrame(columns=['Date'])
for range in season_dt_range:
    tmp = pd.DataFrame(pd.date_range(range[0], range[1], freq='D'),
                       columns=['Date'])
    seasons = seasons.append(tmp)

seasons = seasons[seasons['Date'] > start_data]
seasons.sort_values('Date', inplace=True)
seasons.rename(columns={'Date': 'Season Dates'}, inplace=True)


gamelog_injury = gamelog.merge(injury, left_on=['GAME_DATE', 'PLAYER_NAME'],
                                       right_on=['Date_Injury', 'Player_Injury'], how='outer')


# dropping conflicted rows when the injury data indicated a player sat out when he, in fact, did play
conflict_idx = gamelog_injury.index[gamelog_injury[(gamelog_injury['GAME_DATE'].notnull()) &
                                                   (gamelog_injury['Date_Injury'].notnull())].index]

gamelog_injury.drop(conflict_idx, inplace=True)

# combining the non-nan GAME_DATE and non-nan Date values into one column
GAME_DATE = gamelog_injury['GAME_DATE'].dropna()
date = gamelog_injury['Date_Injury'].dropna()
combined_date = pd.concat([GAME_DATE, date])

# combining the non-nan PLAYER_NAME and non-nan Player values into one column
PLAYER_NAME = gamelog_injury['PLAYER_NAME'].dropna()
Player = gamelog_injury['Player_Injury'].dropna()
combined_player = pd.concat([PLAYER_NAME, Player])

gamelog_injury['DATE'] = combined_date
gamelog_injury['PLAYER'] = combined_player
gamelog_injury.sort_values('DATE', inplace=True)


with open('data/city_distances.pickle', 'r') as f:
    city_distances = pickle.load(f)

players = gamelog_injury['PLAYER'].unique().tolist()

# creating an injury flag column
# a player is marked as injured if an injury occurred in any of the next 3 games
merged = pd.DataFrame()
for player in players:
    tmp = gamelog_injury[gamelog_injury['PLAYER'] == player]
    tmp['INJURED'] = 0
    tmp['MILES_TRAVELED'] = 0

#    tmp['injured'] = tmp.index.apply(lambda i: np.any(tmp['Notes'].iloc[i+1: i+4].notnull())

    for i in xrange(len(tmp) - 1):
        if tmp['Notes'].iloc[i+1: i+4].notnull().sum() > 0:
            tmp['INJURED'].iloc[i] = 1

        city1 = tmp['GAME_LOCATION'].iloc[i]
        city2 = tmp['GAME_LOCATION'].iloc[i+1]
        if city1 == city2:
            pass
        if (city1, city2) in set(city_distances.keys()):
            tmp['MILES_TRAVELED'].iloc[i+1] = city_distances[(city1, city2)]
        elif (city2, city1) in set(city_distances.keys()):
            tmp['MILES_TRAVELED'].iloc[i+1] = city_distances[(city2, city1)]
    merged = merged.append(tmp)


# zero mins played indicate the player played less than one minute.
# Treating those players as not playing
exclude_cols = ['PLAYER', 'DATE', 'INJURED', 'PLAYER_ID', 'NOTES']
merged.ix[merged['MIN'] <= 1.0, merged.columns - exclude_cols] = np.nan
# counting the number of games played in the aggregation window
merged['GAMES_PLAYED'] = merged['MIN'].notnull() * 1

# Create a column to count back to back games
merged["BACK_TO_BACKS"] = 0
for i in xrange(len(merged)-1):
    if merged['DATE'].iloc[i] + pd.DateOffset(days=1) == merged['DATE'].iloc[i+1]:
        merged["BACK_TO_BACKS"].iloc[i+1] = 1
'''
************************TRY*************************
merged.reset_index(drop=True, inplace=True)
merged['BACK_TO_BACKS'] = merged['DATE'].apply(lambda x: 1 if
                    (x.iloc[i] + pd.DateOffset(days=1) == x.loc[i+1]) else 0)
'''

rolling_window = pd.DataFrame()
feat_mat = pd.DataFrame()
for player in players:
    player_df = merged[merged['PLAYER'] == player]
    # maps a player's gamelog to all of the calender dates in a regular season
    player_season = seasons.merge(player_df, left_on='Season Dates',
                                  right_on='DATE', how='left')
    # remove the columns that will not be aggregated
    cols_split = player_season[['Season Dates', 'PLAYER', 'INJURED', 'Notes']]
    # dropping columns that cannot be aggregated
    drop_player_season = ['PLAYER_NAME', 'GAME_DATE', 'YEAR_x', 'YEAR_y',
                          'INJURED', 'MONTH_x', 'MONTH_y', 'DAY_x',
                          'Date_Injury', 'Player_Injury', 'Notes', 'DATE',
                          'DAY_y', 'PLAYER', 'PLAYER_NAME_PACE',
                          'Season Dates', 'PLAYER_NAME_TRACKING', 'GAME_ID',
                          'YEAR_TRACKING', 'YEAR_PACE', 'GAME_ID_TRACKING',
                          'GAME_ID_PACE', 'MATCHUP', 'GAME_LOCATION']
    player_season.drop(drop_player_season, axis=1, inplace=True)
    rolling_window = agg_stats(player_season, window=21)
    # add the aggregated stats and earlier removed columns to feat_mat
    feat_mat = feat_mat.append(pd.concat([rolling_window, cols_split], axis=1))


# drop all nan's from the feature matrix
# the nan's represent games in which the player did not play in
feat_mat.dropna(inplace=True)
feat_mat['START_SEASON'] = feat_mat['Season Dates'].apply(define_season)


# custom apply function to match stats from other dataframes to the feature matrix
# appends the height, weight, and age based on the player and the year
def add_bi_comp_feat(df, player, start_season, feat):
    if df[(df['PLAYER_NAME'] == player) &
              (df['YEAR'] == start_season.year)][feat].empty:
        return None
    return df[(df['PLAYER_NAME'] == player) &
              (df['YEAR'] == start_season.year)][feat].values[0]

feat_mat['HEIGHT'] = 0
feat_mat['HEIGHT'] = feat_mat.apply(lambda x: add_bi_comp_feat(hw_df,
                                                               x['PLAYER'],
                                                               x['START_SEASON'],
                                                               'PLAYER_HEIGHT_INCHES'),
                                   axis=1)

feat_mat['WEIGHT'] = 0
feat_mat['WEIGHT'] = feat_mat.apply(lambda x: add_bi_comp_feat(hw_df,
                                                               x['PLAYER'],
                                                               x['START_SEASON'],
                                                               'PLAYER_WEIGHT'),
                                    axis=1)

feat_mat['AGE'] = 0
feat_mat['AGE'] = feat_mat.apply(lambda x: add_bi_comp_feat(ss_df,
                                                            x['PLAYER'],
                                                            x['START_SEASON'],
                                                            'AGE'),
                                 axis=1)

feat_mat['AGE'] = 0
feat_mat['AGE'] = feat_mat.apply(lambda x: add_bi_comp_feat(ss_df,
                                                            x['PLAYER'],
                                                            x['START_SEASON'],
                                                            'AGE'),
                                 axis=1)