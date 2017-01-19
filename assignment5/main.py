import os
from assignment5.config import basedir
from assignment5.core.nn import NeuralNetwork

TRAINING_FILENAME = os.path.join(basedir, "data/IrisTrainML.dt")
TESTING_FILENAME = os.path.join(basedir, "data/IrisTestML.dt")

NeuralNetwork(TESTING_FILENAME, TRAINING_FILENAME)
