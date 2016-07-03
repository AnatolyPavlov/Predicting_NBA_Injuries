from sklearn.metrics import precision_recall_curve,\
                            roc_curve, roc_auc_score, auc
import numpy as np
import matplotlib as plt


class ROC_PR(object):

    def __init__(self, y_true, probs, which='both'):
        self.y_true = y_true
        self.probs = probs
        self.which = which

    def calc_roc_pr(self):
        self.fpr, self.tpr, self.roc_thres = roc_curve(self.y_true, self.probs)
        random_prob = np.random.random(self.y_true.shape[0])
        self.rand_precision, self.rand_recall, self.rand_thres =\
            precision_recall_curve(self.y_true, random_prob)
        self.precision, self.recall, self.pr_thres =\
            precision_recall_curve(self.y_true, self.probs)
        if self.which == 'both':
            return (self.fpr, self.tpr, self.roc_thres),\
                   (self.precision, self.recall, self.pr_thres)
        elif self.which == 'roc':
            return self.fpr, self.tpr, self.roc_thres
        elif self.which == 'pr':
            return self.precision, self.recall, self.pr_thres

    # plots the roc and precision-recall curve
    def plot_roc_pr(self):

        with plt.style.context('fivethirtyeight'):

            if self.which == 'both':
                self.plot_roc()
                self.plot_pr()
            elif self.which == 'roc':
                self.plot_roc()
            elif self.which == 'pr':
                self.plot_pr()

    def plot_roc(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.fpr, self.tpr)
        # plotting the 45deg random guessing line
        ax.plot(np.linspace(0, 1), np.linspace(0, 1), alpha=.4)
        ax.set_title('ROC CURVE')
        ax.set_xlabel('FALSE POSITIVE RATE')
        ax.set_ylabel('TRUE POSITIVE RATE')
        ax.text(1.1, 1, 'ROC AUC: {:.3f}'.format(roc_auc_score(self.y_true,
                                                               self.probs)))
        ax.show()

    def plot_pr(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.recall, self.precision)
        ax.fill_between(self.recall, 0, self.precision, alpha=.2)
        # plotting a PR-curve from random classification
        ax.plot(self.rand_recall, self.rand_precision, alpha=.4)
        ax.fill_between(self.rand_recall, 0, self.rand_precision, alpha=.2)
        ax.set_title('PRECISION-RECALL CURVE')
        ax.set_xlabel('RECALL')
        ax.set_ylabel('PRECISION')
        ax.text(1.1, 1, 'P-R AUC: {:.3f}'.format(auc(self.recall,
                                                     self.precision)))

def plot_important_features(est, X,
                            save_plot=False,
                            plot_name='feature_importances.png'):
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
        plt.bar(np.arange(X.shape[1]),
                importances[indices],
                color="r",
                alpha=.8,
                yerr=std[indices],
                align="center")
        plt.xticks(np.arange(X.shape[1]),
                   features[indices],
                   rotation='vertical')
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xlabel('FEATURES', fontsize=14)
        plt.ylabel('IMPORTANCES', fontsize=14)
        plt.xlim([-1, X.shape[1]])
        if save_plot:
            plt.savefig('images/{}'.format(plot_name), bbox_inches='tight')