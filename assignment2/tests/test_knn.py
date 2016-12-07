import os
from unittest import TestCase
from assignment2.config import basedir
from assignment2.core.knn import *

TRAINING_FILENAME = os.path.join(basedir, "data/IrisTrainML.dt")
TESTING_FILENAME = os.path.join(basedir, "data/IrisTestML.dt")


class KNNTests(TestCase):
    def test_lambda_double(self):
        training_set = load_file(TRAINING_FILENAME, delimiter=" ")
        training_set.mutate_all_columns(lambda x: float(x))
        training_set.mutate_one_column(1, lambda x: x * 10)
        testing_set = load_file(TESTING_FILENAME, delimiter=" ")
        testing_set.mutate_all_columns(lambda x: float(x))
        assert training_set.len() > 0
        assert testing_set.len() > 0

    def test_euclidean_distance(self):
        e1 = DataEntry([2, 2], 2)
        e2 = DataEntry([4, 4], 0)
        ecd = euclidean_dist(e1, e2)
        self.assertAlmostEqual(ecd, 2.83, places=2)

    def test_get_neighbors(self):
        train_set = DataSet([DataEntry([2, 2], 1), DataEntry([4, 4], 2)])
        test_instance = DataEntry([5, 5], 2)
        neighbors = get_neighbors(train_set.entries, test_instance, 1)
        assert neighbors[0].vals == train_set.entries[1].vals
        assert neighbors[0].ref == train_set.entries[1].ref

    def test_get_response(self):
        neighbors = DataSet([DataEntry([1, 1], 1), DataEntry([2, 2], 1), DataEntry([3, 3], 2)])
        assert get_most_common(neighbors.entries) == 1

    def test_zero_one_loss(self):
        test_set = DataSet([DataEntry([1, 1], 1), DataEntry([2, 2], 1), DataEntry([3, 3], 2)])
        predictions = [1, 1, 1]
        loss = get_zero_one_loss(test_set.entries, predictions)
        self.assertAlmostEqual(33.33, loss, places=2)

    def test_mean(self):
        assert mean([1, 2, 3]) == 2
        assert mean([3, 4, 5]) == 4
        assert mean([3, 4]) == 3.5

    def test_variance(self):
        self.assertAlmostEqual(variance([7, 3, 8]), 2.65, places=2)

    def test_normalize(self):
        ts = DataSet([DataEntry([7, 3, 8], 'a'), DataEntry([1, 2, 3], 'b')])
        normalize(ts, vertical=True)


git init
git add -A
git commit -m "Initialize repository”
git remote add origin git@bitbucket.org:<username>/<repository>.git
git push -u origin —all
