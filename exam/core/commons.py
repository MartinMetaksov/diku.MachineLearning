from math import sqrt, log
import numpy as np
import itertools
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC


def mean(l):
    return sum(l) / len(l)


def variance(l, unbiased=True):
    d = len(l) - 1 if unbiased else len(l)
    m = mean(l)
    return sum(list(map(lambda x: (x - m) ** 2, l))) / d


# Used for regression problems
def mean_squared_error(observed, predicted):
    return sum(list(map(lambda i: (predicted[i] - observed[i]) ** 2, range(len(predicted))))) / len(predicted)


# Used for classification problems
def zero_one_loss(observed, predicted):
    return 100.0 - (sum(map(lambda i: 1 if observed[i] == predicted[i] else 0, range(len(observed)))) /
                    float(len(observed))) * 100.0


# The mean squared error (MSE) with non-linear regression learning (test): 78.3964404
# The mean squared error (MSE) with non-linear regression learning (train):18.2934404
def square_transform(x):
    result = []
    for row in x:
        new_row = []
        for e in row:
            new_row.append(e ** 2)
        for p in itertools.combinations(row, 2):
            new_row.append(sqrt(2) * p[0] * p[1])
        result.append(new_row)
    return np.array(result)


def simple_square_transform(x):
    result = []
    for row in x:
        new_row = []
        for e in row:
            new_row.append(e ** 2)
        result.append(new_row)
    return np.array(result)


def normalize(data, unit_variance=True):
    if len(data) == 0:
        return data
    if unit_variance:
        return np.array(list(map(lambda row: (row - mean(row)) / np.sqrt(variance(row)), data.T))).T
    else:
        return np.array(list(map(lambda row: (row - mean(row)), data.T))).T


def grid_search(c_range, gamma_range, x, y):
    param_grid = dict(gamma=gamma_range, C=c_range)
    cv = KFold(n_splits=5)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(x, y)
    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))
    return None, None
