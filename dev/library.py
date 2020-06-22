import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from sys import stdout
from tick.preprocessing.features_binarizer import FeaturesBinarizer
from tick import linear_model
from time import time


def compute_score(features, features_binarized, labels, blocks_start,
                  blocks_length, C=10, n_folds=10, shuffle=True, n_jobs=1,
                  verbose=False):
    scores = cross_val_score(features, features_binarized, labels, blocks_start,
                             blocks_length, n_folds=n_folds, shuffle=shuffle,
                             C=C, n_jobs=n_jobs, verbose=verbose)

    scores_mean = scores.mean()
    scores_std = scores.std()
    if verbose:
        print("\nscore %0.3f (+/- %0.3f)" % (scores_mean, scores_std))

    return [scores_mean, scores_std]


def cross_val_score(features, features_binarized, labels, blocks_start,
                    blocks_length, n_folds, shuffle, C, n_jobs, verbose):
    cv = KFold(n_splits=n_folds, shuffle=shuffle)
    cv_iter = list(cv.split(features_binarized))

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
    scores = parallel(
        delayed(fit_and_score)(features, features_binarized, labels,
                               blocks_start,
                               blocks_length, idx_train, idx_test, C)
        for (idx_train, idx_test) in cv_iter)
    return np.array(scores)


def fit_and_score(features, features_binarized, labels, blocks_start,
                  blocks_length, idx_train, idx_test, C):
    x_train_bin = features_binarized[idx_train]
    x_test_bin = features_binarized[idx_test]
    y_train, y_test = labels[idx_train], labels[idx_test]

    regr = linear_model.LinearRegression(penalty='binarsity',
                                         blocks_start=blocks_start,
                                         blocks_length=blocks_length,
                                         warm_start=True)
    regr.C = C
    regr.fit(x_train_bin, y_train)
    y_pred = regr.predict(x_test_bin)

    return mean_squared_error(y_test, y_pred)


def get_groups(coeffs):
    n_coeffs = len(coeffs)
    jumps = np.where(coeffs[1:] - coeffs[:-1] != 0)[0] + 1
    jumps = np.insert(jumps, 0, 0)
    jumps = np.append(jumps, n_coeffs)
    groups = np.zeros(n_coeffs)
    for i in range(len(jumps) - 1):
        groups[jumps[i]:jumps[i + 1]] = i
        if jumps[i + 1] - jumps[i] <= 2:
            if i == 0:
                groups[jumps[i]:jumps[i + 1]] = 1
            elif i == len(jumps) - 2:
                groups[jumps[i]:jumps[i + 1]] = groups[jumps[i - 1]]
            else:
                coeff_value = coeffs[jumps[i]]
                group_before = groups[jumps[i - 1]]
                coeff_value_before = coeffs[
                    np.nonzero(groups == group_before)[0][0]]
                try:
                    k = 0
                    while coeffs[jumps[i + 1] + k] != coeffs[
                                        jumps[i + 1] + k + 1]:
                        k += 1
                    coeff_value_after = coeffs[jumps[i + 1] + k]
                except:
                    coeff_value_after = coeffs[jumps[i + 1]]
                if np.abs(coeff_value_before - coeff_value) < np.abs(
                                coeff_value_after - coeff_value):
                    groups[np.where(groups == i)] = group_before
                else:
                    groups[np.where(groups == i)] = i + 1
    return groups.astype(int)


