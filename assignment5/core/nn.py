import csv
import numpy as np
import math
from sklearn import linear_model


class NeuralNetwork:
    def __init__(self, test_file, train_file, delimiter=" "):
        self.test_file = test_file
        self.train_file = train_file
        self.delimiter = delimiter
        raw_train_data = np.matrix(self.open_file(self.train_file))
        raw_test_data = np.matrix(self.open_file(self.test_file))
        indices = raw_train_data[:, 2] != '2'
        raw_train_data = raw_train_data[np.squeeze(indices), :]
        indices = raw_test_data[:, 2] != '2'
        raw_test_data = raw_test_data[np.squeeze(indices), :]
        train_input = raw_train_data[:, 0:2].astype(float)
        train_input_biased = np.insert(train_input, 2, 1, axis=1)
        train_labels = raw_train_data[:, 2].astype(int)
        test_input = raw_test_data[:, 0:2].astype(float)
        np.insert(test_input, 2, 1, axis=1)
        train_label_vals = self.label_dichotomy(train_labels, 1)
        self.compute_gradient(train_input_biased, train_label_vals)
        gradient_descent = np.squeeze(self.gradient_descent(train_input_biased,
                                                            train_label_vals,
                                                            epochs=30,
                                                            threshold=0.01))
        print(gradient_descent)
        best_weight = np.squeeze(self.gradient_descent(train_input_biased,
                                                       train_label_vals,
                                                       epochs=10000,
                                                       threshold=0.01))
        print(gradient_descent)
        logistic_regression = linear_model.LogisticRegression(C=100)
        logistic_regression.fit(train_input_biased, train_label_vals)
        coefficients = np.squeeze(logistic_regression.coef_)
        print(coefficients)
        self.model(train_input_biased[8], best_weight, biased=True)
        self.error(train_input_biased, train_label_vals, best_weight)

    @staticmethod
    def label_dichotomy(dataset_labels, correct, vals=None):
        if vals is None:
            vals = {True: 1, False: -1}
        result = []
        for i in range(len(dataset_labels)):
            check = np.squeeze(np.array(dataset_labels))[i] == correct
            result.append(vals[check])
        return np.array(result)

    @staticmethod
    def compute_gradient(features, labels, weight=None):
        if weight is None:
            weight = np.empty(features.shape[1])
            weight.fill(0.3)
        res_vec = []
        N = len(labels)
        for i in range(N):
            numerator = features[i] * labels[i]
            denominator = 1 + math.exp(labels[i] * (features[i].dot(weight.T).item(0)))
            res_vec.append(numerator / denominator)
        return np.array(-1 / N * (sum(res_vec)))

    def gradient_descent(self, features, labels, epochs=100, learning_rate=0.1, threshold=0.01):
        w_0 = self.compute_gradient(features, labels)
        w_1 = w_0
        for i in range(epochs):
            w_0 = w_1
            gradient = self.compute_gradient(features, labels, weight=w_0)
            gr = []
            for a in np.squeeze(gradient):
                if a > threshold:
                    gr.append(a)
                else:
                    gr.append(0)
            gradient = np.matrix(gr)
            w_1 = w_0 - learning_rate * gradient
            if np.all([a == 0 for a in np.squeeze(gradient)]):
                print("Min. gradient with # of iterations: " + str(i) + ":")
                return w_1
        print("Best input weights with the min. gradient: ")

        return w_1

    @staticmethod
    def model(feature, weights, biased=False):
        if biased:
            s = feature.dot(weights.T).item(0)
        else:
            s = feature.dot(weights[:, :-1].T)
            s = s.item(0) + weights[:, -1].item(0)
        return math.exp(s) / (1 + math.exp(s))

    @staticmethod
    def error(data, labels, weights):
        if len(data) is not len(labels):
            raise ValueError("Inconsistent data length")
        result = []
        n = len(data)
        for i in range(n):
            gradient = math.log(1 + math.exp(-labels[i] * data[i].dot(weights.T)))
            result.append(gradient)
        return sum(result) / n

    def open_file(self, file_name):
        with open(file_name, 'r') as csv_file:
            return list(csv.reader(csv_file, delimiter=self.delimiter))
