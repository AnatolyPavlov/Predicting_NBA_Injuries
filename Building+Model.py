import pandas as pd
import os
import json
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import re
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.grid_search import GridSearchCV
from collections import defaultdict
import cPickle as pickle


# function to extract information from the json files
def json_extract(file):
    with open('data/{}'.format(file)) as f:
        data = json.load(f)
        cols = data['resultSets'][0]['headers']
        vals = data['resultSets'][0]['rowSet']
    return cols, vals


# creates dataframe from the json information
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

    # converting the Date column to datetime objects
    df = parse_date(df, 'Date')

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

    df[df['Player'] == '(William) Tony Parker']['Player'] = 'Tony Parker'
    df = df[df['Player'] != '']

    unique_players = gamelog_df['PLAYER_NAME'].unique()
    for player in unique_players:
        df['Player'] = df['Player'].apply(lambda x:
                                          player if player in x else x)
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


# creates a list of the start and end date of each season
def start_end_season(keyword):

    fns = os.listdir('data/')
    cols = json_extract('2013_{}.json'.format(keyword))[0]

    season_range = []
    for fn in fns:
        if keyword in fn:
            cols, vals = json_extract(fn)
            tmp = pd.DataFrame(vals, columns=cols)
            season_range.append((datetime.strptime(tmp['GAME_DATE'].min(),
                                                   '%Y-%m-%d'),
                                 datetime.strptime(tmp['GAME_DATE'].max(),
                                                   '%Y-%m-%d')))

    del tmp
    seasons = pd.DataFrame(columns=['Date'])
    for rng in season_range:
        tmp = pd.DataFrame(pd.season(rng[0], rng[1], freq='D'),
                           columns=['Date'])
        seasons = seasons.append(tmp)
    return seasons


def create_feat_mat(df, data_window=21):
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


def shuffle_rows(df):
  return df.reindex(np.random.permutation(df.index))


def pace_tracking(keep_cols, keyword):
    df = pickles_to_pandas(keyword, add_year=True)
    df.drop(df.columns - keep_cols, axis=1, inplace=True)
    df.rename(columns={'GAME_ID': 'GAME_ID_{}'.format(keyword.upper()),
                     'PLAYER_NAME': 'PLAYER_NAME_{}'.format(keyword.upper()),
                     'PLAYER_ID': 'PLAYER_ID_{}'.format(keyword.upper()),
                     'YEAR': 'YEAR_{}'.format(keyword.upper())},
              inplace=True)
    return df


# determines the game location by seeing if it's a home (vs.) or away (@) game
def game_loc(x):
    if 'vs.' in x:
        return city_abbrv[re.split(r' vs. ', x)[0]]
    elif '@' in x:
        return city_abbrv[re.split(r' @ ', x)[1]]


# aggregates the stats within a window of specified days
def agg_stats(df, window=21):
    columns = df.columns.tolist()
    cat_dict = defaultdict(list)
    for i in xrange(0, len(df)):
        for col in columns:
            #create a dictionary to store the stats for each category
            cat_dict[col].append(np.nanmean(df[col].iloc[i-window: i]))
        # create a new column that records the number of games played within the window
        cat_dict['GAMES_PLAYED_IN_WINDOW'].append(np.nansum(df['GAMES_PLAYED'].iloc[i-window: i]))
        cat_dict['B2B_PLAYED_IN_WINDOW'].append(np.nansum(df["BACK_TO_BACKS"].iloc[i-window: i]))
        cat_dict['TOTAL_MILES_TRAVELED'].append(np.nansum(df["MILES_TRAVELED"].iloc[i-window: i]))
    return pd.DataFrame(cat_dict)


# defines the start of the season for each row
def define_season(x):
    for season in season_dt_range:
        if (x >= season[0]) & (x <= season[1]):
            return season[0]


# custom apply function to match stats from other dataframes to the feature matrix
# appends the height, weight, and age based on the player and the year
def add_bi_comp_feat(df, player, start_season, feat):
    if df[(df['PLAYER_NAME'] == player) &
              (df['YEAR'] == start_season.year)][feat].empty:
        return None
    return df[(df['PLAYER_NAME'] == player) &
              (df['YEAR'] == start_season.year)][feat].values[0]

# creates dataframes for the season stats, heights and weights,
# gamelogs, and injuries
ss_df = create_df('season_stats', add_year=True)
hw_df = create_df('heights_weights', add_year=True)
gamelog_df = create_df('gamelog')
gamelog_df = prep_gamelog(gamelog_df)
injury_df = pd.read_csv('data/injuries.csv', usecols=[1, 3, 4])
injury_df = prep_injury(injury_df)

# specifies the columns to keep from the pace and tracking data
pace_keep_cols = ['GAME_ID', 'PACE', 'PLAYER_ID', 'PLAYER_NAME', 'YEAR']
tracking_keep_cols = ['GAME_ID', 'SPD', 'DIST', 'PLAYER_ID', 'PLAYER_NAME',
                      'YEAR']
# creates dataframes for the pace and tracking stats
tracking = pace_tracking(tracking_keep_cols, 'tracking')
pace = pace_tracking(pace_keep_cols, 'pace')

# only take data starting from 2013
start_data = datetime.strptime('10-29-2013', '%m-%d-%Y')
gamelog_df = gamelog_df[gamelog['GAME_DATE'] >= start_data]
injury_df = injury_df[injury_df['Date'] >= start_data]
hw_df = hw_df[hw_df['YEAR'] >= 2013]

# renames the injury df columns
injury_df.rename(columns={'Player':'Player_Injury', 'Date': 'Date_Injury'},
                inplace=True)

# merges the pace and tracking data into the gamelogs dataframe
gamelog_df = gamelog_df.merge(pace, left_on=['GAME_ID', 'PLAYER_ID'],
                             right_on=['GAME_ID_PACE', 'PLAYER_ID_PACE'])
gamelog_df = gamelog_df.merge(tracking, left_on=['GAME_ID', 'PLAYER_ID'],
                                right_on=['GAME_ID_TRACKING',
                                          'PLAYER_ID_TRACKING'])

with open('data/city_abbrv.json', 'r') as f:
    city_abbrv = json.load(f)

# identifying the city in which the game took place
gamelog_df['GAME_LOCATION'] = gamelog_df['MATCHUP'].apply(game_loc(x))

# creating dataframe for everyday in a season
seasons = start_end_season('gamelog')
seasons = seasons[seasons['Date'] > start_data]
seasons.sort_values('Date', inplace=True)
seasons.rename(columns={'Date': 'SEASON_DATES'}, inplace=True)

# merging the injury_df to the gamelog_df
gamelog_injury = gamelog_df.merge(injury_df,
                                  left_on=['GAME_DATE', 'PLAYER_NAME'],
                                  right_on=['Date_Injury', 'Player_Injury'],
                                  how='outer')

# dropping conflicted rows when the injury data indicated a player sat out
# when he, in fact, did play
conflict_idx = gamelog_injury.index[gamelog_injury[
                            (gamelog_injury['GAME_DATE'].notnull()) &
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
players = gamelog_injury['PLAYER'].unique().tolist()

# creating an injury flag column
# a player is marked as injured if an injury occurred in the next 3 games
merged = pd.DataFrame()
for player in players:
    tmp = gamelog_injury[gamelog_injury['PLAYER'] == player]
    tmp['INJURED'] = 0
    tmp['MILES_TRAVELED'] = 0
    for i, row in tmp.iterrows():
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

# creating an injury flag column
# a player is marked as injured if an injury occurred in any of the next 3 games
players = gamelog_injury['PLAYER'].unique().tolist()
merged = pd.DataFrame()
for player in players:
    tmp = gamelog_injury[gamelog_injury['PLAYER'] == player]
    tmp['INJURED'] = 0
    tmp['MILES_TRAVELED'] = 0
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

# creating a column counting the number of back to back games
merged["BACK_TO_BACKS"] = merged['DATE'].diff()
merged['BACK_TO_BACKS'] = merged['BACK_TO_BACKS'].apply(lambda x:
                                1 if x == pd.to_timedelta('1 days') else 0)

rolling_window = pd.DataFrame()
feat_mat = pd.DataFrame()
for player in players:
    player_df = merged[merged['PLAYER'] == player]
    # maps a player's gamelog to all of the calender dates in a regular season
    player_season = seasons.merge(player_df, left_on='SEASON_DATES', right_on='DATE', how='left')
    # remove the columns that will not be aggregated
    cols_split = player_season[['SEASON_DATES', 'PLAYER', 'INJURED', 'Notes']]
    # dropping columns that cannot be aggregated
    drop_player_season = ['PLAYER_NAME', 'GAME_DATE', 'YEAR_x', 'YEAR_y', 'INJURED',
                          'MONTH_x', 'MONTH_y', 'DAY_x', 'Date_Injury', 'Player_Injury', 'Notes',
                          'DATE', 'DAY_y', 'PLAYER', 'PLAYER_NAME_PACE', 'SEASON_DATES',
                          'PLAYER_NAME_TRACKING', 'GAME_ID', 'YEAR_TRACKING', 'YEAR_PACE',
                          'GAME_ID_TRACKING', 'GAME_ID_PACE', 'MATCHUP', 'GAME_LOCATION']
    player_season.drop(drop_player_season, axis=1, inplace=True)
    rolling_window = agg_stats(player_season, window=21)
    # add the aggregated stats and the earlier removed columns to the feature matrix
    feat_mat = feat_mat.append(pd.concat([rolling_window, cols_split], axis=1))

feat_mat.drop('Notes', axis=1, inplace=True)

# drop all nan's from the feature matrix
# the nan's represent games in which the player did not play in
feat_mat.dropna(inplace=True)

# convert the weights in the height/weight dataframe to floats
hw_df['PLAYER_WEIGHT'] = hw_df['PLAYER_WEIGHT'].apply(lambda x: float(x))

feat_mat['START_SEASON'] = feat_mat['SEASON_DATES'].apply(define_season)


feat_mat['HEIGHT'] = 0
feat_mat['HEIGHT'] = feat_mat.apply(add_bi_comp_feat(hw_df,
                                                     x['PLAYER'],
                                                     x['START_SEASON'],
                                                     'PLAYER_HEIGHT_INCHES'),
                                   axis=1)

feat_mat['WEIGHT'] = 0
feat_mat['WEIGHT'] = feat_mat.apply(add_bi_comp_feat(hw_df,
                                                     x['PLAYER'],
                                                     x['START_SEASON'],
                                                     'PLAYER_WEIGHT'),
                                    axis=1)

feat_mat['AGE'] = 0
feat_mat['AGE'] = feat_mat.apply(add_bi_comp_feat(ss_df,
                                                  x['PLAYER'],
                                                  x['START_SEASON'],
                                                  'AGE'),
                                 axis=1)

feat_mat['AGE'] = 0
feat_mat['AGE'] = feat_mat.apply(add_bi_comp_feat(ss_df,
                                                  x['PLAYER'],
                                                  x['START_SEASON'],
                                                  'AGE'),
                                 axis=1)

Xy = feat_mat.drop(['PLAYER', 'PLAYER_ID', 'FG3M', 'SEASOn_DATES',
                    'FGM', 'FTM', 'BLK', 'PLUS_MINUS', 'START_SEASON',
                    'PTS', 'REB', 'GAMES_PLAYED', 'PLAYER_ID_PACE',
                    'PLAYER_ID_TRACKING', 'BACK_TO_BACKS', 'PACE', 'MIN',
                    'B2B_PLAYED_IN_WINDOW', 'FGA', 'MILES_TRAVELED'], axis=1)
Xy.dropna(inplace=True)
Xy.reset_index(inplace=True, drop=True)
Xy = Xy[Xy['SPD'] != 0.0]
Xy.rename(columns={'TOTAL_MILES_TRAVELED': 'MILES TRAVELED',
                   'SPD': 'SPEED',
                   'DIST': 'DISTANCE',
                   'AST': 'ASSISTS',
                   'DREB': 'DEF. REBOUNDS',
                   'OREB': 'OFF. REBOUNDS',
                   'PF': 'FOULS',
                   'STL': 'STEALS',
                   'TOV': 'TURNOVERS',
                   'FTA': 'F.T. ATTEMPTED',
                   'FG3A': '3-PTS ATTEMPTED',
                   'GAMES_PLAYED_IN_WINDOW': 'GAMES PLAYED'},
          inplace=True)

y = Xy['INJURED']
X = Xy.drop('INJURED', axis=1)

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=.3)
X_subtrain, X_subtest, y_subtrain, y_subtest = train_test_split(X_train, y_train, test_size=.3)

logreg = LogisticRegressionCV(cv=3).fit(X_subtrain, y_subtrain)
logreg_prob = logreg.predict_proba(X_holdout)[:, 1]
logreg_pred = logreg.predict(X_holdout)

rfc = RandomForestClassifier(n_estimators=250,
                             class_weight='balanced',
                             max_features='log2',
                             n_jobs=-1,
                             oob_score=True).fit(X_subtrain, y_subtrain)
rfc_prob = rfc.predict_proba(X_holdout)[:, 1]
rfc_pred = rfc.predict(X_holdout)


rfc_param_grid = {'n_estimators': [50, 100, 150, 200, 250],
                  'min_samples_leaf': [1, 10, 20, 50],
                  'max_features': ['sqrt', 'log2']}
rfc_gs = GridSearchCV(rfc, param_grid=rfc_param_grid,
                      scoring='roc_auc',
                      cv=5, n_jobs=-1).fit(X_subtrain, y_subtrain)
print rfc_gs.best_params_

gbc = GradientBoostingClassifier(learning_rate=.2,
                                 n_estimators=250,
                                 max_depth=10,
                                 max_features='sqrt').fit(X_subtrain, y_subtrain)
gbc_prob = gbc.predict_proba(X_holdout)[:, 1]
gbc_pred = gbc.predict(X_holdout)

gbc_param_grid = {'learning_rate': [.05, .1, .2, .5],
                  'n_estimators': [50, 100, 150, 200],
                  'max_depth': [1, 3, 6, 10],
                  'max_features': ['auto', 'sqrt']}
gbc_gs = GridSearchCV(gbc, param_grid=gbc_param_grid, scoring='roc_auc', cv=5, n_jobs=-1).fit(X_subtrain, y_subtrain)
gbc_gs.best_params_

def roc_pr(y_true, probs):
    fpr, tpr, thres1 = roc_curve(y_true, probs)
    random_prob = np.random.random(y_true.shape[0])
    random_precision, random_recall, thres2 = precision_recall_curve(y_true, random_prob)
    precision, recall, thres2 = precision_recall_curve(y_true, probs)
    with plt.style.context('fivethirtyeight'):
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(fpr, tpr)
        ax1.fill_between(fpr, 0, tpr, alpha=.2)
        ax1.plot(np.linspace(0, 1), np.linspace(0, 1), alpha=.4, color='r')
        ax1.fill_between(np.linspace(0, 1), 0, np.linspace(0, 1), alpha=.2)
        ax1.set_title('ROC CURVE')
        ax1.set_xlabel('FALSE POSITIVE RATE')
        ax1.set_ylabel('TRUE POSITIVE RATE')
        ax1.text(1.1, 1, 'ROC AUC: {:.3f}'.format(roc_auc_score(y_true, probs)))
        fig1.savefig('images/rfc_roc_curve.png' ,bbox_inches='tight')

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(recall, precision)
        ax2.fill_between(recall, 0, precision, alpha=.2)
        ax2.plot(random_recall, random_precision, alpha=.4)
        ax2.fill_between(random_recall, 0, random_precision, alpha=.2, color='r')
        ax2.set_title('PRECISION-RECALL CURVE')
        ax2.set_xlabel('RECALL')
        ax2.set_ylabel('PRECISION')
        ax2.text(1.1, 1, 'P-R AUC: {:.3f}'.format(auc(recall, precision)))
        fig2.savefig('images/rfc_pr_curve.png', bbox_inches='tight')

roc_pr(y_holdout, rfc_prob)

def plot_important_features(est, X, save_plot=False, plot_name='feature_importances.png'):
    '''
    INPUTS: Model, feature matrix
    OPTIONAL INPUTS: Show a plot, save the plot (BOOLEAN)
    OUTPUT:
    '''
    features = X.columns
    importances = est.feature_importances_
    std = np.std([tree.feature_importances_ for tree in est.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    with plt.style.context('fivethirtyeight'):
        # Plot the feature importances of the forest
        plt.figure(figsize=(14, 8))
        plt.suptitle('FEATURE IMPORTANCES', fontsize=20, y=1.05)
        plt.bar(np.arange(X.shape[1]), importances[indices],
               color="r", alpha=.8, yerr=std[indices], align="center")
        plt.xticks(np.arange(X.shape[1]), features[indices], rotation='vertical')
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('FEATURES', fontsize=14)
        plt.ylabel('IMPORTANCES', fontsize=14)
        plt.xlim([-1, X.shape[1]])
        if save_plot:
            plt.savefig('images/{}'.format(plot_name), bbox_inches='tight')

plot_important_features(rfc, X, True)

with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(12, 10))

    for name, probs in zip(['RANDOM FOREST', 'GRADIENT BOOSTING', 'LOGISTIC REGRESSION'], [rfc_prob, gbc_prob, logreg_prob]):
        fpr, tpr, thres1 = roc_curve(y_holdout, probs)
        random_prob = np.random.random(y_holdout.shape[0])
        random_precision, random_recall, thres2 = precision_recall_curve(y_holdout, random_prob)
        precision, recall, thres2 = precision_recall_curve(y_holdout, probs)
        plt.plot(recall, precision, label=name)
        plt.fill_between(recall, 0, precision, alpha=.2)
        plt.title('PRECISION-RECALL CURVE')
        plt.xlabel('RECALL')
        plt.ylabel('PRECISION')
        plt.legend(fontsize=14)
    plt.savefig('images/combined_curve.png')

thresholds = np.linspace(0, 1, 100)
preds = [rfc_prob > threshold for threshold in thresholds]

f1_scores = [f1_score(y_holdout, pred) for pred in preds]
precision_scores = [precision_score(y_holdout, pred) for pred in preds]
recall_scores = [recall_score(y_holdout, pred) for pred in preds]

with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(14, 12))
    plt.plot(thresholds, f1_scores, label='F1 SCORES')
    plt.plot(thresholds, precision_scores, label='PRECISION SCORES')
    plt.plot(thresholds, recall_scores, label='RECALL SCORES')
    plt.suptitle('F1, PRECISION, AND RECALL VS THRESHOLDS', fontsize=20, y=1.03)
    plt.xlabel('THRESHOLDS')
    plt.ylabel('SCORES')
    plt.legend(loc='best')
    plt.savefig('images/f1_p_r_curves', bbox_inches='tight')

