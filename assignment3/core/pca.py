import csv
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA


class PCATest:
    def __init__(self, filename=None, data=None, classes=None, delimiter=","):
        if data is None:
            self.data = self.load_file(filename, delimiter)
            temp_data = []
            for row in self.data:
                temp_data.append(list(map(float, row)))
            self.data = np.matrix(temp_data)
        else:
            self.data = data
            self.classes = classes

    @staticmethod
    def load_file(filename, delimiter):
        with open(filename, 'r') as csv_file:
            file = list(csv.reader(csv_file, delimiter=delimiter))
            return np.array(file)

    @staticmethod
    def plot_frequency_histogram(data, col_index=-1, label_x="", label_y="", title="", width=0.7):
        data = np.array(data)
        counter = Counter(data[:, col_index])
        counter = OrderedDict(sorted(counter.items(), key=lambda i: (int(i[0]))))
        x = counter.keys()
        vals = list(map(lambda i: i * 100 / len(data), counter.values()))

        indices = np.arange(len(x))
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title(title)
        plt.bar(indices, vals, width)
        plt.xticks(indices + width * 0.5, x)
        plt.show()

    def get_analysis(self):
        X = self.data[:, 0:-1]
        classification = self.data[:, -1]
        n_samples = X.shape[0]

        pca = PCA()
        X_transformed = pca.fit_transform(X)

        X_centered = X - np.mean(X, axis=0)
        cov_matrix = np.dot(X_centered.T, X_centered) / X.shape[0]
        # print("\nCovariance matrix: ")
        # print(cov_matrix)
        return pca.explained_variance_, pca.components_, classification

