import copy
import random
from assignment2.core.knn import *
from unittest import TestCase
from assignment2.core.lr import LinearRegression

TRAINING_FILENAME = "data/IrisTrainML.dt"
TESTING_FILENAME = "data/IrisTestML.dt"
DELIMITER = " "
FOLDS = 5


class MainTests(TestCase):
    # Question 1.1 Nearest neighbors algorithm test
    def test_knn_core(self):
        train_set = load_file(TRAINING_FILENAME, DELIMITER)
        test_set = load_file(TESTING_FILENAME, DELIMITER)
        train_set.mutate_all_columns(lambda x: float(x))
        test_set.mutate_all_columns(lambda x: float(x))
        predict_knn(train_set, test_set, k=1)
        predict_knn(train_set, test_set, k=3)
        predict_knn(train_set, test_set, k=5)

    # Question 1.2 Hyperparameter selection using cross-validation
    def test_cross_validation(self):
        train_set = load_file(TRAINING_FILENAME, DELIMITER)
        train_set.mutate_all_columns(lambda x: float(x))
        self.cross_validation(train_set, FOLDS, "Performance of the 5-Fold cross validation k-NN classifier",
                              "base.png", range(1, 26, 2))

    def cross_validation(self, train_set, folds, title, filename, k_range, print_best=True):
        n = int(len(train_set.entries) / folds)
        chunks = []
        for i in list(range(0, folds)):
            chunks.append(train_set.entries[i * n:(i + 1) * n])
        k_perf = []
        for i in k_range:
            losses = self.zig_zag_invoker(0, folds, chunks, i)
            k_perf.append(sum(losses) / len(losses))
        # print("Average losses per k: " + str(k_perf))
        plot(list(k_range), k_perf, "Choice of k",
             "Average cross-validation loss for k = i", title, target_file=filename)
        k_best_val = min(k_perf)
        k_best = k_range[k_perf.index(k_best_val)]
        print("Average zero-one loss for best_k = " + str(k_best_val) + "%")
        if print_best:
            print("With best_k = " + str(k_best))

    def zig_zag_invoker(self, first, last, chunks, k):
        result = []
        iterator = list(range(first, last))
        iterator.reverse()
        while len(iterator) > 0:
            temp_chunks = copy.copy(chunks)
            test_set = chunks[iterator[0]]
            temp_chunks.remove(chunks[iterator[0]])
            train_set = self.flatten(temp_chunks)
            loss, predictions = predict_knn(DataSet(train_set), DataSet(test_set), k=k, hide=True)
            result.append(loss)
            iterator.remove(iterator[0])
            iterator.reverse()
        return result

    def flatten(self, l):
        return self.flatten(l[0]) + (self.flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

    # Question 1.3 Data Normalization
    def test_normalization(self):
        train_set = load_file(TRAINING_FILENAME, DELIMITER)
        train_set.mutate_all_columns(lambda x: float(x))
        train_set.mutate_one_column(1, lambda x: x * 10)
        normalize(train_set)
        print("k = 3")
        self.cross_validation(train_set, FOLDS, "Performance of cross-validation with normalized train set",
                              "normalized.png", range(1, 26, 2), False)

    def test_linear_regression(self):
        lr = LinearRegression(TRAINING_FILENAME, delimiter=" ")
        for i in range(1, 50):
            print(lr.predict(random.uniform(1.3, 1.9)))
