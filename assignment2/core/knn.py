import csv
import operator
import math
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from assignment2.core.dataset import DataEntry, DataSet


def predict_knn(train_set, test_set, k=1, hide=False):
    predictions = []
    for entry in test_set.entries:
        neighbors = get_neighbors(train_set.entries, entry, k)
        result = get_most_common(neighbors)
        predictions.append(result)

    loss = get_zero_one_loss(test_set.entries, predictions)
    if not hide:
        print_loss(loss, k)
    return loss, predictions


def print_loss(loss, k):
    print('% of incorrect predictions (loss) with k = ' + str(k) + ': ' + str(round(loss, 2)) + '%')


def plot(t, s, x_label, y_label, label, target_file="plot.png", grid=False):
    plt.plot(t, s)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(label)
    plt.grid(grid)
    plt.savefig(target_file)
    # plt.show()


def load_file(filename, delimiter):
    with open(filename, 'r') as csv_file:
        file = list(csv.reader(csv_file, delimiter=delimiter))
        return DataSet(list(map(lambda l: DataEntry(l[0:-1], l[-1]), file)))


def euclidean_dist(e1, e2):
    if len(e1.vals) != len(e2.vals):
        raise ValueError('Lengths of E1 and E2 differ')
    return math.sqrt(sum(list(map(lambda i: pow((e1.vals[i] - e2.vals[i]), 2), range(len(e1.vals))))))


def get_neighbors(ts, ti, k):
    distances = list(map(lambda x: [ts[x], euclidean_dist(ti, ts[x])], range(len(ts))))
    distances.sort(key=operator.itemgetter(1))
    return list(map(lambda x: distances[x][0], range(k)))


def get_most_common(neighbors):
    result = {}
    for neighbor in neighbors:
        result[neighbor.ref] = result[neighbor.ref] + 1 if neighbor.ref in result else 1
    return max(result.__iter__(), key=(lambda key: result[key]))


def get_zero_one_loss(test_set, predictions):
    correct = sum(map(lambda x: 1 if test_set[x].ref == predictions[x] else 0, range(len(test_set))))
    try:
        return 100.0 - (correct / float(len(test_set))) * 100.0
    except ZeroDivisionError:
        print("Cannot divide by 0")


# def normalize(train_set):
#     return preprocessing.normalize(train_set)

def normalize(train_set, vertical=True):
    if train_set.len() == 0:
        return train_set

    ts_vals = np.array(list(map(lambda row: row.vals, train_set.entries)))
    if vertical:
        ts_vals = ts_vals.transpose()

    new_ts_vals = []
    for (i, r) in enumerate(ts_vals):
        m = mean(r)
        print("mean = " + repr(m))
        v = variance(r)
        print("variance = " + repr(v))
        new_row = []
        for (j, rv) in enumerate(r):
            new_row.append(float((rv - m) / v))
        new_ts_vals.append(new_row)

    new_ts_vals = np.array(new_ts_vals)
    if vertical:
        new_ts_vals = new_ts_vals.transpose()

    for (i, e) in enumerate(train_set.entries):
        e.vals = new_ts_vals[i]


def variance(l, unbiased=True):
    d = len(l) - 1 if unbiased else len(l)
    m = mean(l)
    return sqrt(sum(list(map(lambda x: (x - m) ** 2, l))) / d)


def mean(l):
    return sum(l) / len(l)
