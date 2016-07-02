from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import auc, f1_score, precision_score, recall_score,\
                            roc_auc_score, roc_curve, precision_recall_curve
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import numpy as np


def feature_importances(est, X):
    features = X.columns
    importances = est.feature_importances_
    std = np.std([tree.feature_importances_ for tree in est.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # # Print the feature ranking
    # print("Feature ranking:")

    # for f in xrange(X.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    with plt.style.context('fivethirtyeight'):
        # Plot the feature importances of the forest
        plt.figure(figsize=(14, 8))
        plt.suptitle('FEATURE IMPORTANCES', fontsize=20, y=1.1)
        plt.bar(np.arange(X.shape[1]), importances[indices],
               color="r", alpha=.8, yerr=std[indices], align="center")
        plt.xticks(np.arange(X.shape[1]), features[indices],
                   rotation='vertical')
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('FEATURES', fontsize=14)
        plt.ylabel('IMPORTANCES', fontsize=14)
        plt.xlim([-1, X.shape[1]])


Xy = feat_mat.drop(['PLAYER', 'PLAYER_ID', 'FG3M', 'Season Dates',
                    'FGM', 'FTM', 'BLK', 'PLUS_MINUS', 'START_SEASON',
                    'PTS', 'REB', 'GAMES_PLAYED', 'PLAYER_ID_PACE',
                    'PLAYER_ID_TRACKING', 'BACK_TO_BACKS', 'PACE', 'MIN',
                    'GAMES_PLAYED_IN_WINDOW', 'B2B_PLAYED_IN_WINDOW',
                    'HEIGHT', 'FGA', 'MILES_TRAVELED', 'Notes'], axis=1)
Xy.dropna(inplace=True)
Xy.reset_index(inplace=True, drop=True)
# removing all players with zero speed
Xy = Xy[Xy['SPD'] != 0.0]
# renaming the columns to something more meaningful
Xy.rename(columns={'TOTAL_MILES_TRAVELED': 'MILES TRAVELED',
                   'SPD': 'SPEED',
                   'DIST': 'DISTANCE',
                   'AST': 'ASSISTS',
                   'DREB': 'DEFENSIVE REBOUNDS',
                   'OREB': 'OFFENSIVE REBOUNDS',
                   'PF': 'PERSONAL FOULS',
                   'STL': 'STEALS',
                   'TOV': 'TURNOVERS',
                   'FTA': 'FREETHROWS ATTEMPTED',
                   'FG3A': '3-POINTERS ATTEMPTED'},
          inplace=True)

y = Xy['INJURED']
X = Xy.drop('INJURED', axis=1)
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=.3)
X_subtrain, X_subtest, y_subtrain, y_subtest = train_test_split(X_train, y_train, test_size=.3)

logreg = LogisticRegressionCV(cv=3).fit(X_subtrain, y_subtrain)
logreg_prob = logreg.predict_proba(X_holdout)[:, 1]
logreg_pred = logreg.predict(x_holdout)

rfc = RandomForestClassifier(n_estimators=250,
                             class_weight='balanced',
                             max_features='log2',
                             n_jobs=-1,
                             oob_score=True).fit(X_subtrain, y_subtrain)
rfc_prob = rfc.predict_proba(X_holdout)[:, 1]
rfc_pred = rfc.predict(X_holdout)


roc_pr(y_holdout, rfc_prob)

cross_val_score(rfc, X_train, y_train, cv=3, scoring='roc_auc')