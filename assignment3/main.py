import os
from unittest import TestCase
# from assignment3.core.pca import load_file, plot_histogram
from assignment3.config import basedir
from assignment3.core.pca import PCATest
import matplotlib.pyplot as plt

TRAINING_FILENAME = os.path.join(basedir, "data/ML2016TrafficSignsTrain.csv")

shapes = {0: 'o',
          1: 'o',
          2: 'o',
          3: 'o',
          4: 'o',
          5: 'o',
          6: 'o',
          7: 'o',
          8: 'o',
          9: 'o',
          10: 'o',
          11: '^',
          12: 'D',
          13: 'v',
          14: '8',
          15: 'o',
          16: 'o',
          17: 'o',
          18: '^',
          19: '^',
          20: '^',
          21: '^',
          22: '^',
          23: '^',
          24: '^',
          25: '^',
          26: '^',
          27: '^',
          28: '^',
          29: '^',
          30: '^',
          31: '^',
          32: 'o',
          33: 'o',
          34: 'o',
          35: 'o',
          36: 'o',
          37: 'o',
          38: 'o',
          39: 'o',
          40: 'o',
          41: 'o',
          42: 'o',
          }

colors = {0: 'blue',
          1: 'blue',
          2: 'blue',
          3: 'blue',
          4: 'blue',
          5: 'blue',
          6: 'blue',
          7: 'blue',
          8: 'blue',
          9: 'blue',
          10: 'blue',
          11: 'green',
          12: 'yellow',
          13: 'orange',
          14: 'purple',
          15: 'blue',
          16: 'blue',
          17: 'blue',
          18: 'green',
          19: 'green',
          20: 'green',
          21: 'green',
          22: 'green',
          23: 'green',
          24: 'green',
          25: 'green',
          26: 'green',
          27: 'green',
          28: 'green',
          29: 'green',
          30: 'green',
          31: 'green',
          32: 'blue',
          33: 'blue',
          34: 'blue',
          35: 'blue',
          36: 'blue',
          37: 'blue',
          38: 'blue',
          39: 'blue',
          40: 'blue',
          41: 'blue',
          42: 'blue',
          }


class TestMain(TestCase):
    def setUp(self):
        self.pca = PCATest(filename=TRAINING_FILENAME)

    def no_test_plot_histogram(self):
        self.pca.plot_frequency_histogram(self.pca.data, label_x="traffic signs", label_y="class frequencies (in %)",
                                          title="Distribution of class frequencies")

    def test_pca(self):
        eigenvalues, components, classification = self.pca.get_analysis()
        sorted_ev = sorted(eigenvalues, reverse=True)
        print("\n Eigenvalues")
        print(sorted_ev)

        # linear
        x = list(range(0, len(eigenvalues)))
        y = eigenvalues[x]

        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1 - 100, x2 + 1375, y1 - 1, y2 + 4))

        plt.plot(x, y)
        plt.title('eigenspectrum')

        plt.savefig('eigenspectrum.png')

        top_2 = sorted_ev[0:2]
        print("\nFirst two principal components")
        print(top_2)
        temp_sum = 0.0
        total_sum = sum(eigenvalues)
        for i, ev in enumerate(sorted_ev):
            temp_sum += ev
            explanation = temp_sum * 100 / total_sum
            if explanation > 90:
                print("\nWith " + str(i + 1) + " components, the variance is " + str(explanation) + "% explained")
                break

        plt.figure()
        # plt.scatter(x_axis, y_axis)
        plt.title('Scatter of the first 2 components')
        for (i, component) in enumerate(components):
            label = "sign" + str(i)
            plt.plot(component[0], component[1], marker=shapes.get(int(classification[i])),
                     color=colors.get(int(classification[i])), label=label)
            # plt.legend()
        plt.savefig('scatter.png')
