import csv
import os
from exam.config import basedir
import numpy as np


def load_data(filename, data_type=float, delimiter=","):
    with open(os.path.join(basedir, filename), 'r') as csv_file:
        file = list(csv.reader(csv_file, delimiter=delimiter))
        return np.array(file, dtype=data_type)
