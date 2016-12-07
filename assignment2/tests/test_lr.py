import os
import random
from unittest import TestCase
from assignment2.config import basedir
from assignment2.core.lr import LinearRegression

TRAINING_FILENAME = os.path.join(basedir, "data/DanWood.dt")


class LRTests(TestCase):
    def test_lr_class(self):
        lr = LinearRegression(TRAINING_FILENAME, delimiter=" ")
        self.assertAlmostEqual(lr.w[0], 9.49, places=2)
        self.assertAlmostEqual(lr.b, -10.43, places=2)
        print(lr.hat_matrix)

    def test_predict(self):
        lr = LinearRegression(TRAINING_FILENAME, delimiter=" ")
        self.assertAlmostEqual(lr.predict(1.700)[1], 5.70, places=2)

    def test_multiple_predictions(self):
        lr = LinearRegression(TRAINING_FILENAME, delimiter=" ")
        for i in range(1, 6):
            lr.predict(random.uniform(1.3, 1.9))
        assert len(lr.predictions) == 5

    def test_plot(self):
        lr = LinearRegression(TRAINING_FILENAME, delimiter=" ")
        for i in range(1, 50):
            lr.predict(random.uniform(1.3, 1.9))
        lr.plot("lr.png")

    def test_mean_squared_error(self):
        lr = LinearRegression(TRAINING_FILENAME, delimiter=" ")
        mse = lr.mean_squared_error()
        print("MSE : " + repr(mse))
        self.assertAlmostEqual(mse, 0.01, places=2)

    def test_hat_matrix_is_symmetric(self):
        lr = LinearRegression(TRAINING_FILENAME, delimiter=" ")
        h_m = lr.hat_matrix
        h_m_t = h_m.transpose()
        assert h_m[0][0] == h_m_t[0][0]
        assert h_m[0][1] == h_m_t[1][0]

