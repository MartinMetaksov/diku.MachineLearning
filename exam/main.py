from unittest import TestCase
from exam.core.commons import variance, mean_squared_error, zero_one_loss, simple_square_transform, normalize
from exam.core.regression import LinearRegression, NonlinearRegression
from exam.utility.file_utility import load_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

SPECTROSCOPIC_REDSHIFTS_TRAIN_FILE = "data/ML2016SpectroscopicRedshiftsTrain.dt"
SPECTROSCOPIC_REDSHIFTS_TEST_FILE = "data/ML2016SpectroscopicRedshiftsTest.dt"
ESTIMATED_REDSHIFTS_TRAIN_FILE = "data/ML2016EstimatedRedshiftsTrain.dt"
ESTIMATED_REDSHIFTS_TEST_FILE = "data/ML2016EstimatedRedshiftsTest.dt"
GALAXIES_TRAIN_FILE = "data/ML2016GalaxiesTrain.dt"
GALAXIES_TEST_FILE = "data/ML2016GalaxiesTest.dt"
WEED_CROP_TRAIN = "data/ML2016WeedCropTrain.csv"
WEED_CROP_TEST = "data/ML2016WeedCropTest.csv"


class GalaxyDistances(TestCase):
    def setUp(self):
        self.spectroscopic_redshifts_train = load_data(SPECTROSCOPIC_REDSHIFTS_TRAIN_FILE)
        self.spectroscopic_redshifts_test = load_data(SPECTROSCOPIC_REDSHIFTS_TEST_FILE)
        self.estimated_redshifts_train = load_data(ESTIMATED_REDSHIFTS_TRAIN_FILE)
        self.estimated_redshifts_test = load_data(ESTIMATED_REDSHIFTS_TEST_FILE)
        self.galaxies_train = load_data(GALAXIES_TRAIN_FILE)
        self.galaxies_test = load_data(GALAXIES_TEST_FILE)
        self.red_shifts_variance_train = variance(self.spectroscopic_redshifts_train)
        self.mse = mean_squared_error(self.spectroscopic_redshifts_test, self.estimated_redshifts_test)
        # According to Christian Igel's announcement during the exam
        self.correct_weights = np.array(
            [0.0185134, 0.0479647, -0.0210943, -0.0274002, -0.0226798, 0.0064449, 0.0151842, 0.0120738, 0.0103486,
             0.00599684, -0.0294513, 0.069059, 0.00630583, -0.00472042, -0.00873932, 0.00311043, 0.0017252, 0.00435176]
        )
        self.correct_offset = -0.801881

    # Question 1.1 (Data preparation). Compute the variance σ2 of the red-shifts in the training data
    # ML2016SpectroscopicRedshiftsTrain.dt. Compute the error of the SDSS predictions on the test data
    # (i.e., how good the predictions in ML2016EstimatedRedshiftsTest.dt match the data in
    # ML2016SpectroscopicRedshiftsTest.dt) using the mean squared error.
    # Deliverables: Report variance of redshifts in the training data, report error of SDSS predictions on test data
    def test_data_preparation(self):
        print("Question 1.1 Data Preparation")
        print("Variance of redshifts in the train data: %.7f" % self.red_shifts_variance_train)
        print("The mean squared error (MSE): of the SDSS predictions on the test data: %.7f" % self.mse)

    # Question 1.2 (Linear regression). Apply linear regression to the data. Train on the training data
    # (ML2016GalaxiesTrain.dt and ML2016SpectroscopicRedshiftsTrain.dt) and evaluate the model on the test data
    # (ML2016GalaxiesTest.dt and ML2016SpectroscopicRedshiftsTest.dt). Report the model and its quality measured by
    # the mean squared error on training and test set.
    #
    # Divide the obtained mean squared error by the variance σ2 computed in question red
    # 1.1. What does the result tell you? In general, what would a result smaller or larger than one tell you?
    def test_linear_regression(self):
        print("Question 1.2 Linear regression")
        model = LinearRegression()
        model.fit(self.galaxies_train, self.spectroscopic_redshifts_train)
        # However, as provided in the announcement during the exam, the input features are dependent and the
        # weights obtained from the model are incorrect. Therefore, changing the weights and the offset:
        model.w = self.correct_weights
        model.b = self.correct_offset
        predictions_test = model.predict(self.galaxies_test)
        predictions_train = model.predict(self.galaxies_train)
        mse_test = mean_squared_error(predictions_test, self.spectroscopic_redshifts_test)
        mse_train = mean_squared_error(predictions_train, self.spectroscopic_redshifts_train)
        print("The mean squared error (MSE) after the linear regression learning (test set): %.7f" %
              mse_test)
        print("The mean squared error (MSE) after the linear regression (train set): %.7f" %
              mse_train)
        normalized_error = self.mse / self.red_shifts_variance_train
        print("The normalized error is : %.7f" % normalized_error)
        variance_pred = variance(predictions_test)
        normalized_error_predictions = mse_test / variance_pred
        nse = 1 - normalized_error_predictions
        print("The normalized error of the predictions is : %.7f" % normalized_error_predictions)
        print("The Nash-Sutcliffe Efficiency of the predictions is : %.7f" % nse)

    # Question 1.3 (Non-linear regression). Now apply a non-linear regression method to the data. Note that you may
    # only use the training data for model selection and hyperparameter tuning. Please describe your approach and your
    # results in detail.
    # Train on the training data set and evaluate on the test data set. Discuss the results in comparison to the
    # linear regression model. You are free to apply any non-linear method you like. You have to briefly argue why you
    # selected a particular method. If you choose a method that was not introduced in the course, you are supposed to
    # describe the algorithm in full detail.
    def test_nonlinear_regression(self):
        print("Question 1.3 Non-linear regression")
        model = NonlinearRegression(simple_square_transform)
        model.fit(self.galaxies_train, self.spectroscopic_redshifts_train)
        predictions_test = model.predict(self.galaxies_test)
        mse = mean_squared_error(predictions_test, self.spectroscopic_redshifts_test)
        print("The mean squared error (MSE) with non-linear regression learning (test): "
              "%.7f" % mse)
        predictions_train = model.predict(self.galaxies_train)
        mse_train = mean_squared_error(predictions_train, self.spectroscopic_redshifts_train)
        print("The mean squared error (MSE) with non-linear regression learning (train):"
              "%.7f" % mse_train)


class Weeds(TestCase):
    def setUp(self):
        self.weeds_train = load_data(WEED_CROP_TRAIN)
        self.weeds_test = load_data(WEED_CROP_TEST)
        self.x_train = self.weeds_train[:, :13]
        self.y_train = self.weeds_train[:, -1]
        self.x_test = self.weeds_test[:, :13]
        self.y_test = self.weeds_test[:, -1]

    # Question 2.1 (Logistic Regression). Train a logistic regression model on the training data set and evaluate it
    # on the test set. Report the model parameters. Measure the classification errors using the 0-1 loss.
    def test_logistic_regression(self):
        print("Question 2.1 Logistic regression")
        model = LogisticRegression()

        model.fit(self.x_train, self.y_train)
        weights_coefficients = model.coef_[0]
        intercept_offset_bias = model.intercept_[0]
        print("The parameters of the logistic regression model: \nweights = " + str(weights_coefficients))
        print("\nintercept (offset) = " + str(intercept_offset_bias))
        predictions_test = model.predict(self.x_test)
        zol_test = zero_one_loss(self.y_test, predictions_test)
        print("The zero one loss after logistic regression on the test data is: %.5f" % zol_test)
        # predictions_train = model.predict(self.x_train)
        # zol_train = zero_one_loss(self.y_train, predictions_train)
        # print("The zero one loss after logistic regression on the train data is: %.5f" % zol_train)

    # Question 2.2 (Binary classification using support vector machines). Description of software used; a short
    # description of how you pro- ceeded; initial γ or σ value suggested by Jaakkola’s heuristic; optimal C and γ
    # found by grid search; classification accuracy on training and test data
    def test_binary_classification_with_svm(self):
        print("Question 2.2 Binary classification using SVMs")
        # model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        #             decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        #             max_iter=-1, probability=False, random_state=None, shrinking=True,
        #             tol=0.001, verbose=False)
        model = SVC()
        model.fit(self.x_train, self.y_train)
        predictions_test = model.predict(self.x_test)
        zol_test = zero_one_loss(self.y_test, predictions_test)
        print("The zero one loss after logistic regression on the test data is: %.5f" % zol_test)
        # todo: fix SVM
        # predictions_train = model.predict(self.x_train)
        # zol_train = zero_one_loss(self.y_train, predictions_train)
        # print("The zero one loss after logistic regression on the train data is: %.5f" % zol_train)

    # Question 2.3 (normalization)
    def test_normalization(self):
        print("Question 2.3 Normalization")
        x = normalize(self.weeds_train)
        # todo: test SVM with normalized data

    # Question 2.4 (Principal component analysis)
    def test_pca(self):
        print("Question 2.4 PCA")
        pca = PCA()
        x_transform = pca.fit_transform(self.x_train)
        eigenvalues = pca.explained_variance_
        # eigenspectrum
        print("\n Eigenvalues")
        print(eigenvalues)
        x = list(range(0, len(eigenvalues)))
        y = eigenvalues[x]
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1 - 1, x2 + 13, y1 - 100000, y2 + 3500000))
        plt.plot(x, y)
        plt.title('eigenspectrum')
        plt.savefig('eigenspectrum.png')
        # first 2 principal components
        top_2 = eigenvalues[0:2]
        print("\nFirst two principal components")
        print(top_2)
        temp_sum = 0.0
        total_sum = sum(eigenvalues)
        for i, ev in enumerate(eigenvalues):
            temp_sum += ev
            explanation = temp_sum * 100 / total_sum
            if explanation > 90:
                print("\nWith " + str(i + 1) + " components, the variance is " + str(explanation) + "% explained")
                break

        # scatter
        plt.figure()
        x0, y0, x1, y1 = [], [], [], []
        for (i, row) in enumerate(x_transform[:, : 2]):
            if int(self.y_train[i]) == 0:
                x0.append(row[0])
                y0.append(row[1])
            else:
                x1.append(row[0])
                y1.append(row[1])
        plt.scatter(x0, y0, color='r', marker='x', alpha=.4, label="weed")
        plt.scatter(x1, y1, color='b', marker='o', alpha=.4, label="crop")
        plt.legend()
        plt.title('Scatter of the first 2 components')
        plt.savefig('scatter.png')

    # Question 2.5 (2-means clustering) Perform 2-means clustering of ML2016WeedCropTrain.csv. For the submission,
    # initialize the cluster centers with the first two data points in ML2016WeedCropTrain.csv (that is not a
    # recommended intialization technique, but makes it easier to corrrect the exam). Plot the cluster centers
    # projected to the first two principal components. That is, add the cluster centers to the plot from the previous
    # question. Briefly discuss the results: Did you get meaningful clusters?
    def test_clustering(self):
        print("Question 2.5 Clustering")
        pca = PCA()
        x_transform = pca.fit_transform(self.x_train)

        custom_init_centers = np.array([self.x_train[0], self.x_train[1]])
        km = KMeans(n_clusters=2, init=custom_init_centers)
        km.fit_predict(x_transform)

        centers_1 = km.cluster_centers_[0]
        centers_2 = km.cluster_centers_[1]

        # scatter
        plt.figure()
        x0, y0, x1, y1 = [], [], [], []
        for (i, row) in enumerate(x_transform[:, : 2]):
            if int(self.y_train[i]) == 0:
                x0.append(row[0])
                y0.append(row[1])
            else:
                x1.append(row[0])
                y1.append(row[1])
        plt.scatter(x0, y0, color='r', marker='x', alpha=.4, label="weed")
        plt.scatter(x1, y1, color='b', marker='o', alpha=.4, label="crop")
        plt.plot(centers_1[0], centers_1[1], color='c', marker='*', alpha=1)
        plt.plot(centers_2[0], centers_2[1], color='c', marker='*', alpha=1, label="centers")
        plt.legend()
        plt.title('Scatter of the first 2 components with cluster centers')
        plt.savefig('kmeans_scatter.png')
