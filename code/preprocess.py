import pandas as pd
import os
import json
from datetime import datetime
import numpy as np
import re
from collections import defaultdict
import cPickle as pickle


# function to extract information from the json files
def json_extract(filename):
    with open('../data/{}'.format(filename)) as f:
        data = json.load(f)
        cols = data['resultSets'][0]['headers']
        vals = data['resultSets'][0]['rowSet']
    return cols, vals


# creates dataframe from the json information
def create_df(keyword, add_year=False):
    '''
    INPUT: keyword = 'gamelog', 'season_stats', or 'heights_weights'
           The add_year option adds a YEAR column that has a value associated
           with the year from the filename.
    OUTPUT: dataframe
    '''
    fns = os.listdir('../data/')
    cols = json_extract('../data/2013_{}.json'.format(keyword))[0]
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


# converts the date column into a datatime object
def parse_date(df, date_col, create_sep_cols=True):
    '''
    INPUT: dataframe, the name of the column to be converted to datetime
           The create_sep_cols option splits the date into YEAR, MONTH, and
           DAY columns.
    OUTPUT: dataframe
    '''
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    if create_sep_cols:
        date = df[date_col]
        df['YEAR'] = date.apply(lambda x: x.year)
        df['MONTH'] = date.apply(lambda x: x.month)
        df['DAY'] = date.apply(lambda x: x.day)
    return df


# used in an apply function.
# removes things like (a) and (b)
def clean_notes(x):
    found = re.findall(r'\s\(\w\)', x)
    if found:
        return x.replace(found[0], '')
    else:
        return x


# preprocess injury df
def prep_injury(df):

    # converting the Date column to datetime objects
    df = parse_date(df, 'Date')

    # only take data starting from 2013
    df = df[df['Date'] >= start_data]

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
                                                if re.match(r'\w\.\w\.', x)
                                                else x)

    # removing characters like (a) and (b)
    df['Player'] = df['Player'].apply(lambda x: ' '.join(x.split()[:2])
                                                if re.match(r'.+\(.+\)', x)
                                                else x)

    # manually correcting Tony Parker's name
    df[df['Player'] == '(William) Tony Parker']['Player'] = 'Tony Parker'
    df = df[df['Player'] != '']

    # compare names in the injury list with names from the nba
    for player in players:
        df['Player'] = df['Player'].apply(lambda x:
                                          player if player in x else x)

    # renames the injury df columns
    df.rename(columns={'Player': 'Player_Injury', 'Date': 'Date_Injury'},
              inplace=True)

    return df


# preprocess gamelogs
def prep_gamelog(df):

    # converts the Date column to datetime objects
    df = parse_date(df, 'GAME_DATE')

    # drop useless columns
    drop_vars = ['SEASON_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'WL', 'FG_PCT',
                 'FG3_PCT', 'FT_PCT', 'VIDEO_AVAILABLE']
    df.drop(drop_vars, axis=1, inplace=True)
    return df


# dropping conflicted rows when the injury data indicated a player sat out
# when he, in fact, did play
def remove_conflicts(df):
    conflict_idx = df.index[df[(df['GAME_DATE'].notnull()) &
                               (df['Date_Injury'].notnull())].index]
    df.drop(conflict_idx, inplace=True)
    return df

# combining the non-nan GAME_DATE and non-nan Date values into one column
def combine_non_nan(df):
    GAME_DATE = df['GAME_DATE'].dropna()
    date = df['Date_Injury'].dropna()
    combined_date = pd.concat([GAME_DATE, date])
    df['DATE'] = combined_date

    # extract the non-nan PLAYER_NAME and non-nan Player values into new column
    df['PLAYER'] = df[['PLAYER_NAME', 'Player_Injury']].fillna('').sum(axis=1)
    return df


# creates a list of the start and end date of each season
def find_season_beg_end():

    fns = os.listdir('../data/')
    cols = json_extract('../data/2013_gamelog.json')[0]

    season_beg_end = []
    for fn in fns:
        if 'gamelog' in fn:
            cols, vals = json_extract(fn)
            tmp = pd.DataFrame(vals, columns=cols)
            season_beg_end.append((datetime.strptime(tmp['GAME_DATE'].min(),
                                                     '%Y-%m-%d'),
                                   datetime.strptime(tmp['GAME_DATE'].max(),
                                                     '%Y-%m-%d')))
    return season_beg_end


# creates a dataframe with all of the possible dates within a season
def create_season_dates():
    season_beg_end = find_season_beg_end()
    season_dates = pd.DataFrame(columns=['Date'])
    for rng in season_beg_end:
        tmp = pd.DataFrame(pd.date_range(rng[0], rng[1], freq='D'),
                           columns=['Date'])
        season_dates = season_dates.append(tmp)

    season_dates.sort_values('Date', inplace=True)
    season_dates.rename(columns={'Date': 'SEASON_DATES'}, inplace=True)
    return season_dates


# creates the feature matrix
# the default aggregation window is 21 days
def create_feat_mat(df, data_window=21):
    feat_mat = pd.DataFrame()
    player_ids = df['PLAYER_ID'].unique().tolist()
    for player_id in player_ids:
        tmp = df[df['PLAYER_ID'] == player_id]
        roll_mean_df = pd.rolling_mean(tmp, data_window)
        feat_mat = feat_mat.append(roll_mean_df)
    return feat_mat


# converts pickled files to dataframes
def pickles_to_pandas(keyword, add_year=False):

    fns = os.listdir('../data/')

    with open('../data/2014_{}_stats.pickle'.format(keyword), 'r') as f:
        pkl_f = pickle.load(f)
        cols = pkl_f[0]['resultSets'][0]['headers']
    df = pd.DataFrame(columns=cols)

    for fn in fns:
        if keyword in fn:
            with open('../data/{}'.format(fn), 'r') as f:
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


# adds pace and tracking stats to the feature matrix
def pace_tracking(keep_cols, keyword):
    '''
    INPUT: a list of columns from the pace and tracking dataframes to keep.
           keyword = 'pace' or 'tracking'
    OUTPUT: dataframe
    '''
    df = pickles_to_pandas(keyword, add_year=True)
    df.drop(df.columns - keep_cols, axis=1, inplace=True)
    df.rename(columns={'GAME_ID': 'GAME_ID_{}'.format(keyword.upper()),
                       'PLAYER_NAME': 'PLAYER_NAME_{}'.format(keyword.upper()),
                       'PLAYER_ID': 'PLAYER_ID_{}'.format(keyword.upper()),
                       'YEAR': 'YEAR_{}'.format(keyword.upper())},
              inplace=True)
    return df


# adds an INJURY and MILES TRAVELED column to the dataframe
def add_injury_miles_traveled(df, players):
    '''
    INPUT: the dataframe to add these columns to, a list of all the players
           in the NBA
    OUTPUT: dataframe
    '''
    with open('../data/city_distances.pickle', 'r') as f:
        city_distances = pickle.load(f)

    # creating an injury flag column
    # a player is marked as injured if an injury occurred in the next 3 games
    merged = pd.DataFrame()
    for player in players:
        tmp = df[df['PLAYER'] == player]
        tmp['INJURED'] = 0
        tmp['MILES_TRAVELED'] = 0
        for i in xrange(len(tmp)-1):
            if tmp['Notes'].iloc[i+1: i+4].notnull().sum() > 0:
                tmp['INJURED'].iloc[i] = 1

            city1 = tmp['GAME_LOCATION'].iloc[i]
            city2 = tmp['GAME_LOCATION'].iloc[i+1]
            if city1 == city2:
                pass
            if (city1, city2) in set(city_distances.keys()):
                tmp['MILES_TRAVELED'].iloc[i+1] = city_distances[(city1,
                                                                  city2)]
            elif (city2, city1) in set(city_distances.keys()):
                tmp['MILES_TRAVELED'].iloc[i+1] = city_distances[(city2,
                                                                  city1)]
        merged = merged.append(tmp)
    return merged


# determines the game location by seeing if it's a home (vs.) or away (@) game
def game_loc(x):
    with open('../data/city_abbrv.json', 'r') as f:
        city_abbrv = json.load(f)
    if 'vs.' in x:
        return city_abbrv[re.split(r' vs. ', x)[0]]
    elif '@' in x:
        return city_abbrv[re.split(r' @ ', x)[1]]


# counting the number of games played in the aggregation window
def count_games(df):
    df['GAMES_PLAYED'] = df['MIN'].notnull() * 1
    return df


# creating a column counting the number of back to back games
def add_b2b(df):
    df["BACK_TO_BACKS"] = df['DATE'].diff()
    df['BACK_TO_BACKS'] = df['BACK_TO_BACKS']\
                            .apply(lambda x:1 if x == pd.to_timedelta('1 days')
                                              else 0)
    return df


# aggregates the stats within a window of specified days
def agg_stats(df, window):
    columns = df.columns.tolist()
    cat_dict = defaultdict(list)
    for i in xrange(0, len(df)):
        for col in columns:
            # create a dictionary to store the stats for each category
            cat_dict[col].append(np.nanmean(df[col].iloc[i-window: i]))
        # create a new column that records the number of games played within
        # the window
        cat_dict['GAMES_PLAYED_IN_WINDOW'].append(np.nansum(
                                        df['GAMES_PLAYED'].iloc[i-window: i]))
        cat_dict['B2B_PLAYED_IN_WINDOW'].append(np.nansum(
                                        df["BACK_TO_BACKS"].iloc[i-window: i]))
        cat_dict['TOTAL_MILES_TRAVELED'].append(np.nansum(
                                        df["MILES_TRAVELED"].iloc[i-window: i]))
    return pd.DataFrame(cat_dict)


def moving_window(df, players, seasons):
    aggregated = pd.DataFrame()
    feat_mat = pd.DataFrame()
    for player in players:
        player_df = df[df['PLAYER'] == player]
        # maps players' gamelogs to the calender dates in a regular season
        player_season = seasons.merge(player_df, left_on='SEASON_DATES',
                                      right_on='DATE', how='left')
        # remove the columns that will not be aggregated
        cols_split = player_season[['SEASON_DATES', 'PLAYER', 'INJURED',
                                    'Notes']]
        # dropping columns that cannot be aggregated
        drop_player_season = ['PLAYER_NAME', 'GAME_DATE', 'YEAR_x', 'YEAR_y',
                              'INJURED', 'MONTH_x', 'MONTH_y', 'DAY_x',
                              'Date_Injury', 'Player_Injury', 'Notes', 'DATE',
                              'DAY_y', 'PLAYER', 'PLAYER_NAME_PACE',
                              'SEASON_DATES', 'PLAYER_NAME_TRACKING',
                              'GAME_ID', 'YEAR_TRACKING', 'YEAR_PACE',
                              'GAME_ID_TRACKING', 'GAME_ID_PACE', 'MATCHUP',
                              'GAME_LOCATION']
        player_season.drop(drop_player_season, axis=1, inplace=True)
        aggregated = agg_stats(player_season)
        # add the aggregated stats and the earlier removed columns to the
        # feature matrix
        feat_mat = feat_mat.append(pd.concat([aggregated, cols_split], axis=1))
    return feat_mat


# custom apply function to match stats from other df's to the feature matrix
def add_feat_to_df(df, player, start_season, feat):
    if df[(df['PLAYER_NAME'] == player) &
          (df['YEAR'] == start_season.year)][feat].empty:
        return None
    return df[(df['PLAYER_NAME'] == player) &
              (df['YEAR'] == start_season.year)][feat].values[0]


# defines the start of the season for each row
def define_season(x, season_beg_end):
    for season in season_beg_end:
        if (x >= season[0]) & (x <= season[1]):
            return season[0]


def add_heights_weights(df, season_beg_end):
    df['START_SEASON'] = df['SEASON_DATES']\
                         .apply(lambda x: define_season(x, season_beg_end))
    for cat in ['PLAYER_HEIGHT_INCHES', 'PLAYER_WEIGHT']:
        df[cat] = df.apply(lambda x:
                           add_feat_to_df(hw_df, x['PLAYER'],
                                                 x['START_SEASON'], cat),
                           axis=1)
    df.rename(columns={'PLAYER_HEIGHT_INCHES': 'HEIGHT',
                       'PLAYER_WEIGHT': 'WEIGHT'}, inplace=True)
    return df


def add_age(df):
    df['AGE'] = df.apply(lambda x: add_feat_to_df(ss_df, x['PLAYER'],
                                   x['START_SEASON'], 'AGE'), axis=1)
    return df


if __name__ == '__main__':
    # start the data from the first day of the 2013 season
    start_data = datetime.strptime('10-29-2013', '%m-%d-%Y')

    # creates dataframes for the season stats, heights and weights
    ss_df = create_df('season_stats', add_year=True)
    hw_df = create_df('heights_weights', add_year=True)
    # convert the weights in the height/weight dataframe to floats
    hw_df['PLAYER_WEIGHT'] = hw_df['PLAYER_WEIGHT'].apply(lambda x: float(x))

    # creates dataframes for the gamelogs and injuries
    gamelog_df = create_df('gamelog')
    gamelog_df = prep_gamelog(gamelog_df)
    players = gamelog_df['PLAYER_NAME'].unique().tolist()
    injury_df = pd.read_csv('../data/injuries.csv', usecols=[1, 3, 4])
    injury_df = prep_injury(injury_df)

    # specifies the columns to keep from the pace and tracking data
    pace_keep_cols = ['GAME_ID', 'PACE', 'PLAYER_ID', 'PLAYER_NAME', 'YEAR']
    tracking_keep_cols = ['GAME_ID', 'SPD', 'DIST', 'PLAYER_ID', 'PLAYER_NAME',
                          'YEAR']

    # creates dataframes for the pace and tracking stats
    tracking = pace_tracking(tracking_keep_cols, 'tracking')
    pace = pace_tracking(pace_keep_cols, 'pace')

    # merges the pace and tracking data into the gamelogs dataframe
    gamelog_df = gamelog_df.merge(pace, left_on=['GAME_ID', 'PLAYER_ID'],
                                  right_on=['GAME_ID_PACE',
                                            'PLAYER_ID_PACE'])
    gamelog_df = gamelog_df.merge(tracking, left_on=['GAME_ID', 'PLAYER_ID'],
                                  right_on=['GAME_ID_TRACKING',
                                            'PLAYER_ID_TRACKING'])

    # identifying the city in which the game took place
    gamelog_df['GAME_LOCATION'] = gamelog_df['MATCHUP'].apply(game_loc)

    # creating dataframe for everyday in a season
    season_dates = create_season_dates()
    season_dates = season_dates[season_dates['SEASON_DATES'] > start_data]

    # merging the injury_df to the gamelog_df
    gamelog_injury = gamelog_df.merge(injury_df,
                                      left_on=['GAME_DATE',
                                               'PLAYER_NAME'],
                                      right_on=['Date_Injury',
                                                'Player_Injury'],
                                      how='outer')

    gamelog_injury = remove_conflicts(gamelog_injury)

    gamelog_injury = combine_non_nan(gamelog_injury)

    gamelog_injury.sort_values('DATE', inplace=True)

    merged = add_injury_miles_traveled(gamelog_injury, players)

    # zero mins played indicate the player played less than one minute.
    # Treating those players as not playing
    exclude_cols = ['PLAYER', 'DATE', 'INJURED', 'PLAYER_ID', 'NOTES']
    merged.ix[merged['MIN'] <= 1.0, merged.columns - exclude_cols] = np.nan

    merged = count_games(merged)
    merged = add_b2b(merged)

    feat_mat = moving_window(merged, players, season_dates)
    feat_mat.drop('Notes', axis=1, inplace=True)

    # drop all nan's from the feature matrix
    # the nan's represent games in which the player did not play in
    feat_mat.dropna(inplace=True)

    season_beg_end = find_season_beg_end()
    feat_mat = add_heights_weights(feat_mat, season_beg_end)
    feat_mat = add_age(feat_mat)

    # average running speed of 0 likely means an error in the data logging or
    # the player played very little
    feat_mat = feat_mat[feat_mat['SPD'] != 0.0]

    with open('../pickles/feat_mat.pickle', 'w') as f:
        pickle.dump(feat_mat, f)