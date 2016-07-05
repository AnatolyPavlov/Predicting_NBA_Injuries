from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier,\
                             GradientBoostingClassifier,\
                             ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from sklearn.grid_search import GridSearchCV
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def main():

    with open('../pickles/feat_mat.pickle', 'r') as f:
        feat_mat = pickle.load(f)

    drop_feats = ['PLAYER', 'PLAYER_ID', 'FG3M', 'Season Dates',
                  'FGM', 'FTM', 'BLK', 'PLUS_MINUS', 'START_SEASON',
                  'PTS', 'REB', 'GAMES_PLAYED', 'PLAYER_ID_PACE',
                  'PLAYER_ID_TRACKING', 'BACK_TO_BACKS', 'PACE', 'MIN',
                  'B2B_PLAYED_IN_WINDOW', 'FGA', 'MILES_TRAVELED']

    feat_mat = feature_select(feat_mat, drop_feats)

    X, y = create_Xy(feat_mat)

    # splits data into a training set and a holdout set
    # the holdout set is what will be used to report the final score
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y,
                                                              test_size=.3)
    # splits the training set into a sub-training set and a sub-test set
    # the sub-test set will be used to compare models
    X_subtrain, X_subtest, y_subtrain, y_subtest = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=.3)

    # train a logistic regression model
    logreg = LogisticRegressionCV(cv=3).fit(X_subtrain, y_subtrain)
    logreg_prob = logreg.predict_proba(X_subtest)[:, 1]
    logreg_pred = logreg.predict(X_subtest)

    # train a random forest model
    # used gridsearch to select the best parameters
    rfc = RandomForestClassifier(n_estimators=250,
                                 class_weight='balanced',
                                 max_features='log2',
                                 n_jobs=-1,
                                 oob_score=True).fit(X_subtrain, y_subtrain)
    rfc_prob = rfc.predict_proba(X_subtest)[:, 1]

    # train a gradient boosted tree model
    # used gridsearch to select the best parameters
    gbc = GradientBoostingClassifier(learning_rate=.2,
                                     n_estimators=250,
                                     max_depth=10,
                                     max_features='sqrt')\
                                    .fit(X_subtrain, y_subtrain)
    gbc_prob = gbc.predict_proba(X_subtest)[:, 1]

    # train an extra trees model
    # used gridsearch to select the best parameters
    etc = ExtraTreesClassifier(n_estimators=200, n_jobs=-1)\
                              .fit(X_subtrain, y_subtrain)
    etc_prob = etc.predict_proba(X_subtest)[:, 1]

#    probs_list = [etc_prob, rfc_prob, logreg_prob, gbc_prob]
#    model_names = ['EXTRA TREES', 'RANDOM FOREST', 'LOGISTIC REGRESSION',
#                   'GRADIENT BOOSTED TREES']
    probs_list = [etc_prob]
    model_names = ['EXTRA TREES']
    plot_roc_pr(y_subtest, probs_list, model_names, save_plot=True)
    plot_important_features(etc, X_subtest, save_plot=True,
                            plot_name='feature_importances.png')

    for model, prob in zip(model_names, probs_list):
        print '{} Precision-Recall AUC: {}'.format(model, pr_auc(y_subtest,
                                                                 prob))

    with open('../pickles/etc_model.pickle', 'w') as f:
        pickle.dump(etc, f)


# select which feature to use in the feature matrix
def feature_select(df, drop_feats):

    # remove features
    df = df.drop(drop_feats, axis=1)

    # NAN's are the first data points that were not aggregated
    df.dropna(inplace=True)

    # reseting the df index
    df.reset_index(inplace=True, drop=True)

    # renaming the features to something more meaningful
    df.rename(columns={'TOTAL_MILES_TRAVELED': 'MILES TRAVELED',
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
    return df


# splits the full matrix into the feature matrix and target variables
def create_Xy(df):
    # separates the matrix into the features and target variables
    y = df['INJURED']
    X = df.drop('INJURED', axis=1)
    return X, y


# grid search for the best hyperparameters
def search_best_params(X, y, est, search_params, scoring='roc_auc', cv=3):
    '''
    INPUT:
        X: feature matrix
        y: target variables
        est: model
        search_params: a dictionary of parameters to grid search
        scoring: how to score the cross validations (default is roc_auc)
        cv: the number of cross validations (default is 3)
    OUTPUT: best parameters
    '''
    gs = GridSearchCV(est, param_grid=search_params, scoring='roc_auc',
                      cv=5, n_jobs=-1).fit(X, y)
    return gs.best_params_


# plots the ROC and precision-recall curves of all the models on one plot
def plot_roc_pr(y_true, probs_list, model_names, save_plot=False):
    '''
    INPUT: y_true, a list of probabilities, a list of model names
           Provide a list of multiple probabilities and model names to compare
           plots.
           Provide a list of a single probability and model name for only one
           plot.
    OUTPUT: plots
    '''
    with plt.style.context('fivethirtyeight'):
        fig1, ax1 = plt.subplots(figsize=(12, 10))
        for model, prob in zip(model_names, probs_list):
            precision, recall, thres1 = precision_recall_curve(y_true, prob)
            ax1.plot(recall, precision, label=model)
            ax1.fill_between(recall, 0, precision, alpha=.2)
            ax1.set_title('PRECISION-RECALL CURVE')
            ax1.set_xlabel('RECALL')
            ax1.set_ylabel('PRECISION')
        random_prob = np.random.random(y_true.shape[0])
        random_precision, random_recall, thres2 =\
                                precision_recall_curve(y_true, random_prob)
        ax1.plot(random_recall, random_precision, alpha=.5, c='k',
                label='RANDOM GUESSING')
        plt.legend(loc='best')
        if save_plot:
            fig1.savefig('../images/all_pr_curve.png', bbox_inches='tight')

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        for model, prob in zip(model_names, probs_list):
            fpr, tpr, thres2 = roc_curve(y_true, prob)
            ax2.plot(fpr, tpr, label=model)
            ax2.fill_between(fpr, 0, tpr, alpha=.2)
            ax2.set_title('ROC CURVE')
            ax2.set_xlabel('FALSE POSITIVE RATE')
            ax2.set_ylabel('TRUE POSITIVE RATE')
        ax2.plot(np.linspace(0, 1), np.linspace(0, 1), alpha=.5, c='k')
        ax2.fill_between(np.linspace(0, 1), 0, np.linspace(0, 1), alpha=.2)
        plt.legend(loc='best')
        if save_plot:
            fig2.savefig('../images/all_roc_curve.png', bbox_inches='tight')


# plots the feature importances
def plot_important_features(est, X, save_plot=False,
                            plot_name='feature_importances.png'):
    '''
    INPUTS: Model, feature matrix
    OPTIONAL INPUTS: Show a plot, save the plot (BOOLEAN)
    OUTPUT:
    '''
    features = X.columns
    importances = est.feature_importances_
    std = np.std([tree.feature_importances_ for tree in est.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    with plt.style.context('fivethirtyeight'):
        # Plot the feature importances of the forest
        plt.figure(figsize=(16, 8))
        plt.suptitle('FEATURE IMPORTANCES', fontsize=20, y=1.05)
        plt.bar(np.arange(X.shape[1]), importances[indices],
                color="r", yerr=std, alpha=.8, align="center")
        plt.xticks(np.arange(X.shape[1]), features[indices],
                   rotation='vertical')
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('FEATURES', fontsize=14)
        plt.ylabel('IMPORTANCES', fontsize=14)
        plt.xlim([-1, X.shape[1]])
        if save_plot:
            plt.savefig('../images/{}'.format(plot_name), bbox_inches='tight')


# calculates the area under the precision-recall curve
def print_scores(y_true, prob):
    precision, recall, thres = precision_recall_curve(y_true, prob)
    return auc(recall, precision)

if __name__ == '__main__':

    main()
