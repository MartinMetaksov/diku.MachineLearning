from unittest import TestCase
from assignment3.core.pca import PCA
import numpy as np

data = np.matrix('1, 2, 3, 4, 5; 6, 7, 8, 9, 10; 11, 12, 13, 14, 15')


class TestPCA(TestCase):
    def setUp(self):
        self.pca = PCA(data=data)

    def test_shape(self):
        assert data.shape == (3, 5)

    def test_mean_vector(self):
        assert self.pca.get_analysis()[0] == 3.0
        assert self.pca.get_analysis()[1] == 8.0
        assert self.pca.get_analysis()[2] == 13.0

    def test_covariance_matrix(self):
        print(self.pca.get_covariance_matrix())

        # Covariance Matrix:
        #  [[ 0.68814855  0.14632437 -0.10886832]
        #  [ 0.14632437  1.36691503  0.47695815]
        #  [-0.10886832  0.47695815  1.04965928]]
