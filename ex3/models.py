import numpy as np
import sklearn as skl

import matplotlib.pyplot as plt
# import plotly.graph_objects as go
from plotnine import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from numpy import linalg
import time


"""remark: m = number of samples from X, d = number of features of X or  labels of y (2)
          """

def type_of_conclusion(y_real, y_predict):
    TN, FP, TP, FN = 0, 0, 0, 0
    for i in range(len(y_real)):
        # if y_predict[i] == 0: FP += 1
        # the label can be only  +-1, so if one of them is 0, i decided it will be false positive cuz its to hard to  abel it as true
        if y_real[i] == y_predict[i] and y_predict[i] == -1: TN += 1
        elif y_real[i] == y_predict[i] and y_predict[i] == 1: TP += 1
        elif y_real[i] != y_predict[i] and y_predict[i] == -1: FN += 1
        elif y_real[i] != y_predict[i] and y_predict[i] == 1: FP += 1

    return TN, FP, TP, FN

def score_helper(X, y_real, y_predict):
    TN, FP, TP, FN = type_of_conclusion(y_real, y_predict)
    P = np.count_nonzero(y_real == 1)  # number of positive in the real label
    N = np.count_nonzero(y_real == -1)  # number of negative in the real label
    error_rate = (FP + FN) / (P + N)
    accuracy = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    specificity = TN / N
    return{"num samples": len(y_real), "error": error_rate, "accuracy": accuracy,
           "FPR": FP / N, "TPR": TP / P, "precision": precision, "specificity": specificity}




class Perceptron:

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        """
        learns the parameters of the model and stores the trained model
        :param X: training set as X m x d training set
        :param y: label y {-1, +1}^m
        :return: void
        """
        x_dim = X.shape[0]
        X_with__intercept = np.insert(X, 0, np.ones(x_dim), axis=1)  # adding 1 to the first col of X
        w = np.zeros(X_with__intercept.shape[1])  # creating the model vector init with 0  of d + 1 cols
        while np.any(np.sign(X_with__intercept @ w) - y):
            #if not exist yi<w,xi> <=0 <-> sign the model w gave to X is wrong (!=y)
            #  output  not zero then the wil wont beak
            wronged_labeled = self._get_any_wrong_labeled(X_with__intercept, y, w)
            xi, yi = wronged_labeled[0], wronged_labeled[1]
            w = w + xi * yi
        self.model = w

    def predict(self, X):
        """
        predicts the label of each sample by sign<w,xi>
        :param X: unlabeled set X m' x d
        :return:label y dim m', yi from {-1, +1},
        """
        x_dim = X.shape[0]
        X_with_intercept = np.insert(X, 0, np.ones(x_dim), axis=1)  # adding 1 to the first row of X
        return np.sign(X_with_intercept @ self.model)  # the prediction

    def score(self, X, y):
        """
        calculate fields:
        num samples: number of samples in the test set
        error: error (misclassification) rate
        accuracy: accuracy
        FPR: false positive rate
        TPR: true positive rate
        precision: precision
        specificty: specificty
        :param X: unlabeled test sex X m' x d
        :param y: ture labels y = {-1,+1}^m' of this (x) test set
        :return: a dictionary with the the above fields
        """

        y_predict = self.predict(X)
        return score_helper(X, y, y_predict)


    def _get_any_wrong_labeled(self, X, y, w):
        """
        find the xi (row vector) in the training set which the classifer predicted wrong (!= y)
        :param X: training set X m+1 x d
        :param y: label value +-1
        :param w: model to train
        :return: tuple(xi, yi) xi: of of the rows form the training X set with the wrong prediction
                               yi: the wrong labeled
                if not found, return -1
        """
        x_dim = X.shape[0]
        for row in range(x_dim):
            if y[row] * (X[row] @ w) <= 0:
                return X[row], y[row]
        return -1


class LDA:
    """
    assumptions: yi ~ Ber(p), xi|yi ~ N(mu, sigma), same sigma aka cov matrix to all
                and there's a solution for the classifier which is linear separable
    """
    def __init__(self):
        self.probability_of_y = None  # [pr_y1, pr_y_minus1]
        self.mu_y = None  # [mu_y1, mu_y_minus1]
        self.cov_matrix_sigma = None
        self.cov_matrix_sigma_inverse = None
        self.bias = None
        self.model_param = (self.probability_of_y, self.mu_y, self.cov_matrix_sigma_inverse)

    def fit(self, X, y):
        pr_y1 = (1 / len(y)) * (np.count_nonzero(y == 1))  # 1/n sum(1[yi=1])
        pr_y_minus1 = (1 / len(y)) * (np.count_nonzero(y == -1))  # 1/n sum(1[yi=-1])
        self.probability_of_y = np.array([pr_y1, pr_y_minus1])

        # splitting the samples between rows labeled 1 and rows labeled -1, for each
        # group we found we calculate the mean (sum each col values and dived by number of rows)
        # each sample have 2 cols, so the we wll get mean vector with 2 coordinates
        x_labeled_1 = X[y == 1, :]
        x_labeled_minus1 = X[y == -1, :]
        mu_y1 = x_labeled_1.mean(axis=0)
        mu_y_minus1 = x_labeled_minus1.mean(axis=0)
        self.mu_y = np.array([mu_y1, mu_y_minus1])

        # sigma = cov(X.T) should be k x 2 dim cuz later ned to mult my muy is vector with 2 coordinates
        self.cov_matrix_sigma = np.cov(X.T)

        # we assume the cov matrix comes from the normal distribution so she have an inverse
        self.cov_matrix_sigma_inverse = np.linalg.inv(self.cov_matrix_sigma)

        # -(1/2)muy.T * sigma^-1 muy + ln(Pr(y))
        x_labeled_1_bais = (-0.5 * (self.mu_y[0, :].T @ self.cov_matrix_sigma_inverse @ self.mu_y[0, :]) \
                            + np.log(self.probability_of_y[0]))
        x_labeled_minus1_bais = (-0.5 * (self.mu_y[1, :].T @ self.cov_matrix_sigma_inverse @ self.mu_y[1, :]) \
                                  + np.log(self.probability_of_y[1]))
        self.bias = np.array([x_labeled_1_bais, x_labeled_minus1_bais])

    def predict(self, X):
        """
        prediction rule: argmax(deltay) = argmax xT*sigma^-1 * muy + bias
        where bias = -0.5 * mnuyT * sigmna^-1 *muy + ln(Pr(y))
        :param X: sample to predict
        :return: prediction label y dim m', yi from {-1, +1},
        """
        y_predict = np.argmax(X @ self.cov_matrix_sigma_inverse @ self.mu_y + self.bias, axis=1)
        # arg max will return in this case labels 0 1 (0 index for 1 and 1 index for 1 in the _muy array)
        # so we need to fix it to be 1 for index 0 and -1 for index 1
        return -2 *y_predict + 1

    def score(self, X, y):
        y_predict = self.predict(X)
        b = score_helper(X, y, y_predict)
        return b

class SVM:

    def __init__(self):
        self.svm = SVC(C=1e10, kernel="linear")
        self.intercept = None
        self.model = None

    def fit(self, X,y):
        self.svm.fit(X, y)
        self.model = self.svm.coef_[0]
        self.intercept = self.svm.intercept_[0]

    def predict(self, X):
        return self.svm.predict(X)

    def score(self, X, y):
        y_predict = self.predict(X)
        return score_helper(X, y, y_predict)

class Logistic:

    def __init__(self):
        self.logistic = LogisticRegression(solver='liblinear')
        self.model = None
        self.intercept = None

    def fit(self, X, y):
        self.logistic.fit(X, y)
        self.model = self.logistic.coef_[0]
        self.intercept = self.logistic.intercept_[0]

    def predict(self, X):
        return self.logistic.predict(X)

    def score(self, X, y):

        y_predict = self.predict(X)
        return score_helper(X, y, y_predict)


class DecisionTree:
    def __init__(self):
        self.decisionTree = DecisionTreeClassifier(min_samples_split=5, max_depth=7)
        self.model = None
        self.intercept = None

    def fit(self, X, y):
        self.decisionTree.fit(X, y)
        self.model = self.decisionTree.tree_

    def predict(self, X):
        return self.decisionTree.predict(X.T)

    def score(self, X, y):
        y_predict = self.predict(X)
        return score_helper(X, y, y_predict)


if __name__ == '__main__':
    p = Perceptron()
    X = np.array([[-3, -2, 2, 3, 3, -3], [1, 1.5, -2, -2, -3, 3]]).T
    y_1 = np.array([1, -1, -1, 1, 1, -1])
    p.fit(X, y_1)
    p.score(X, y_1)
    # print("asdasd\n", p.predict(X))

    # print(p.model)
    # print(y_1)
    # print(p.predict(X))
