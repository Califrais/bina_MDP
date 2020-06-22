import numpy as np
from tick import linear_model
import time
from tick.preprocessing.features_binarizer import FeaturesBinarizer
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import pandas as pd
from dev.library import compute_score, get_groups

def Q_value(Q_params, action, x):
    """Recovers the value at point x of the discrete action-value fct component Q (,l)
     from its definition with the cut-points and the weights.

    Parameters:
        Q_params : list of list, shape = (nb-actions, 6)
                    each list is the return of the function binarsity_reg 2 applied to the the learning sample
                    {(s_i, q_l(s_i) ; a_i=l}. We will only use the info about the cut-points and final_coeffs.

        action : 'int'
                    the action (that we also denote l) in {1,...,L} for which we compute the discrete
                    action-value fct component

        x :    numpy.ndarray, shape = (dimension of observation space)
    """
    if Q_params == []:
        return 0

    cut_points = Q_params[action][0]
    intercept = Q_params[action][1]["intercept"]
    coeffs = Q_params[action][1]["weights"]

    X_bin = []
    for i in range(len(x)):
        for j, cut in enumerate(cut_points[str(i) + ":continuous"][1:]):
            if x[i] < cut:
                X_bin.append(1.)
                X_bin += [0. for i in range(len(cut_points[str(i) + ":continuous"]) - (j + 2))]
                break
            else:
                X_bin.append(0.)

    return intercept + np.dot(coeffs, X_bin)


def binarsity_reg(X, y, grid_C=np.logspace(-2, 2, 10), C=None, verbose=True):
    """implementation of the linear regression on binarized features with binarsity penalty.
    Parameters:
        X : pandas dataframe, shape=(1420, 4), columns=[ str(j)+":continuous" for j in range(len(X[0]))], dtype='float'
            list of continuous features

        y :numpy.ndarray, dtype='float'
            list of labels

        grid_C : list or numpy.ndarray, dtype='float'
        this list of weigths associated to the binarsity penalty from which will be chosen
                the final by cross-validation if C=None

        C : float (positive) or None
            weigth associated to the binarsity penalty

        verbose: Boolean
                    If True, prints additional info

    Returns :
        cut_points_estimates : dict, length = dimension of observation space

        final_coeffs : numpy.ndarray

        blocks_start : numpy.ndarray

        all_groups : list

        coeffs : numpy.ndarray

        regr.C : 'float'

    """

    t0 = time.clock()
    if verbose == True:
        print("binarsity regression number of observations :", len(X))
    n_cuts = 50
    binarizer = FeaturesBinarizer(n_cuts=n_cuts, detect_column_type="column_names")
    X_bin = binarizer.fit_transform(X)

    features_names = [X.columns[j] for j in range(len(X.columns))]
    boundaries = binarizer.bins_boundaries

    blocks_start = binarizer.blocks_start
    blocks_length = binarizer.blocks_length

    n_folds = 5

    if C == None:
        scores_cv = pd.DataFrame(columns=['C', 'scores_mean', 'scores_std'])
        for i, C_i in enumerate(grid_C):
            scores = compute_score(X, X_bin, y, blocks_start,
                                   blocks_length, C=C_i, n_folds=n_folds)
            scores = [C_i] + scores
            scores_cv.loc[i] = scores
        if verbose == True:
            print("cross_val scores :")
            print(scores_cv.round(3))

        idx_min = scores_cv.scores_mean.argmin()
        C_best = grid_C[idx_min]

        idx_chosen = min([i for i, j in enumerate(list(scores_cv.scores_mean <= scores_cv.scores_mean.min()
                                                       + scores_cv.scores_std[idx_min])) if j])
        C_chosen = grid_C[idx_chosen]
        if verbose == True:
            print("C_best :", "%.4g" % C_best)
            print("C_chosen :", "%.4g" % C_chosen)

    regr = linear_model.LinearRegression(penalty='binarsity',
                                         blocks_start=blocks_start,
                                         blocks_length=blocks_length,
                                         warm_start=True)
    if C == None:
        regr.C = C_chosen
    else:
        regr.C = C
    if verbose == True:
        print("regr.C :", "%.4g" % regr.C)

    regr.fit(X_bin, y)
    coeffs = regr.weights

    # computations of the cut-points
    all_groups = list()
    cut_points_estimates = {}
    for j, start in enumerate(blocks_start):
        coeffs_j = coeffs[start:start + blocks_length[j]]
        all_zeros = not np.any(coeffs_j)
        if all_zeros:
            cut_points_estimate_j = np.array([-np.inf, np.inf])
            groups_j = np.array(blocks_length[j] * [0])
        else:
            groups_j = get_groups(coeffs_j)
            # print("group"+str(j), groups_j)
            jump_j = np.where(groups_j[1:] - groups_j[:-1] != 0)[0] + 1
            if jump_j.size == 0:
                cut_points_estimate_j = np.array([-np.inf, np.inf])
            else:
                cut_points_estimate_j = boundaries[features_names[j]][
                    jump_j]
                if cut_points_estimate_j[0] != -np.inf:
                    cut_points_estimate_j = np.insert(cut_points_estimate_j,
                                                      0, -np.inf)
                if cut_points_estimate_j[-1] != np.inf:
                    cut_points_estimate_j = np.append(cut_points_estimate_j,
                                                      np.inf)
        cut_points_estimates[features_names[j]] = cut_points_estimate_j
        if j > 0:
            groups_j += max(all_groups) + 1
        all_groups += list(groups_j)

    if verbose == True:
        print("cutpoints :")
        for j in range(len(cut_points_estimates)):
            print(features_names[j], ["%.4f" % cut_points_estimates[features_names[j]][i] for i in
                                      range(len(cut_points_estimates[features_names[j]]))])

    # creation of final binarized X data for the computed cutpoints
    binarizer2 = FeaturesBinarizer(method='given', bins_boundaries=cut_points_estimates)
    X_bin2 = binarizer2.fit_transform(X)
    X_bin2 = np.array(X_bin2.todense())
    blocks_start2 = binarizer2.blocks_start
    blocks_length2 = binarizer2.blocks_length
    X_bin2_train, X_bin2_test, y_train, y_test = train_test_split(X_bin2, y, test_size=0.2)

    # final re-fit:
    regr3 = linear_model.LinearRegression(penalty='binarsity', blocks_start=blocks_start2, blocks_length=blocks_length2,
                                          warm_start=True)
    regr3.C = 1e10
    regr3.fit(X_bin2_train, y_train)
    if verbose == True:
        print("R² score of final predictor on train data (80% of total data) :",
              "%.4g" % regr3.score(X_bin2_train, y_train))
        print("R² score of final predictor on test data (20% of total data) :",
              "%.4g" % regr3.score(X_bin2_test, y_test))

    final_coeffs = {"intercept": regr3.intercept, "weights": regr3.weights}

    t1 = time.clock()
    if verbose == True:
        print("time elapsed for binarsity regression step:", "%.4g" % (t1 - t0), "s")

    return cut_points_estimates, final_coeffs, blocks_start, all_groups, coeffs, regr.C