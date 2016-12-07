from matplotlib.pyplot import scatter, title, xlabel, ylabel, show
from numpy import *
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, filename, delimiter=','):
        self.dataset = loadtxt(filename, delimiter=delimiter)
        self.x = self.dataset[:, 0]
        self.y = self.dataset[:, 1]
        self.x_tilde = np.matrix(list(map(lambda i: [i, 1], self.x)))
        self.x_tilde_t = self.x_tilde.transpose()
        self.hat_matrix = np.array((inv(self.x_tilde_t * self.x_tilde) * self.x_tilde_t).dot(self.y))
        self.w = np.array(self.hat_matrix[0][:-1])
        self.b = self.hat_matrix[0][-1]
        self.predictions = []

    def model(self, x):
        return self.w * x + self.b

    def predict(self, x):
        self.predictions.append([x, self.model(x)[0]])
        return self.predictions[-1]

    def mean_squared_error(self):
        predictions = list(map(lambda i: self.predict(i), self.x))
        total = 0
        for i, p in enumerate(predictions):
            total += (p[1] - self.y[i]) ** 2
        return total / len(predictions)

    def plot(self, target_file):
        x = self.x

        for row in self.predictions:
            x = append(row[0], x)

        y = self.model(x)
        plt.plot(x, self.model(x), '-')
        plt.plot(x, y, '.')
        plt.title('Carbon filament lamp linear regression')
        plt.xlabel('Absolute temperature (in 1000 degrees Kelvin)')
        plt.ylabel('Energy radiation per cm2 per second')
        plt.savefig(target_file)
        # plt.show()
