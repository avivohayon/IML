import matplotlib.pyplot as plt
import numpy as np
# from comparison import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import time

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_images = np.logical_or((y_train == 0), (y_train == 1))
test_images = np.logical_or((y_test == 0), (y_test == 1))
x_train, y_train = x_train[train_images], y_train[train_images]
x_test, y_test = x_test[test_images], y_test[test_images]


def draw_three():
    """
    draw 3 images samples labeled with '0' and 3 with '1'
    """
    plt.imshow(x_test[0])
    plt.show()
    plt.imshow(x_test[5])
    plt.show()
    plt.imshow(x_test[2])
    plt.show()
    plt.imshow(x_test[1])
    plt.show()
    plt.imshow(x_test[3])
    plt.show()
    plt.imshow(x_test[6])
    plt.show()


def rearrange_data(X):
    """
    given data as a array of size nˆ28ˆ28, returns
    a new matrix of size nˆ784 with the same data
    :param X:
    :return: new matrix of size nˆ784 with the same data
    """
    return X.reshape(X.shape[0], 784)

def test_and_compare(x_test, x_train, y_test, y_train):
    m = np.array([50, 100, 300, 500])
    logistic_mean_acc = np.array([])
    svm_mean_acc = np.array([])
    tree_mean_acc = np.array([])
    k_near_neighb_mean_acc = np.array([])

    logistic_mean_elapsed_time = np.array([])
    svm_mean_elapsed_time = np.array([])
    tree_mean_elapsed_time = np.array([])
    k_near_neighb_mean_elapsed_time = np.array([])


    k = 50
    for i in m:
        print(i)
        logistic_acc, svm_acc, tree_acc, k_near_acc = 0, 0, 0, 0
        logistic_elapsed, svm_elapsed, tree_elapsed, k_near_elapsed = 0, 0, 0, 0

        for j in range(k):
            random_indexes = np.random.choice(x_train.shape[0], i, replace=False)
            x_random_train_samples = x_train[random_indexes]
            y_random_train_samples = y_train[random_indexes]
            while (1 not in y_random_train_samples) or (0 not in y_random_train_samples):
                random_indexes = np.random.choice(x_train.shape[0], i)
                x_random_train_samples = x_train[random_indexes]
                y_random_train_samples = y_train[random_indexes]

            start_time = time.time()
            logistic = LogisticRegression()
            logistic.fit(x_random_train_samples, y_random_train_samples)
            logistic_acc += logistic.score(x_test, y_test)
            logistic_elapsed += (time.time() - start_time)

            start_time = time.time()
            svm = SVC()
            svm.fit(x_random_train_samples, y_random_train_samples)
            svm_acc += svm.score(x_test, y_test)
            svm_elapsed += (time.time() - start_time)

            start_time = time.time()
            decision_tree = DecisionTreeClassifier()
            decision_tree.fit(x_random_train_samples, y_random_train_samples)
            tree_acc += decision_tree.score(x_test, y_test)
            tree_elapsed += (time.time() - start_time)

            start_time = time.time()
            k_nearest =KNeighborsClassifier()
            k_nearest.fit(x_random_train_samples, y_random_train_samples)
            k_near_acc += k_nearest.score(x_test, y_test)
            k_near_elapsed += (time.time() - start_time)



        # the mean has to be in range of (0, 1) so we dived by the number of k = iteration 50
        logistic_mean_acc = np.append(logistic_mean_acc, logistic_acc / k)
        svm_mean_acc = np.append(svm_mean_acc, svm_acc / k)
        tree_mean_acc = np.append(tree_mean_acc, tree_acc / k)
        k_near_neighb_mean_acc = np.append(k_near_neighb_mean_acc, k_near_acc / k)
        print("logistic_acc = ", logistic_acc)
        print("svm acc = ", svm_acc)
        print("tree_acc = ", tree_acc)
        print("k_near_acc = ", k_near_acc)


        logistic_mean_elapsed_time = np.append(logistic_mean_elapsed_time, logistic_elapsed / k)
        svm_mean_elapsed_time = np.append(svm_mean_elapsed_time, svm_elapsed / k)
        tree_mean_elapsed_time = np.append(tree_mean_elapsed_time, tree_elapsed / k)
        k_near_neighb_mean_elapsed_time = np.append(k_near_neighb_mean_elapsed_time, k_near_elapsed / k)
        print("logistic_mean_elapsed_time ", logistic_mean_elapsed_time)
        print("svm_mean_elapsed_time", svm_mean_elapsed_time)
        print("tree_mean_elapsed_time = ", tree_mean_elapsed_time)
        print("k_near_neighb_mean_elapsed_time", k_near_neighb_mean_elapsed_time)



    plt.plot(m, logistic_mean_acc, color="purple")
    plt.plot(m, svm_mean_acc, color="green")
    plt.plot(m, tree_mean_acc, color="orange")
    plt.plot(m, k_near_neighb_mean_acc, color="blue")

    plt.title("accuracy for different number of samples: Logistic, SVM,Decision Tree,K Nearest Neighbors")
    plt.legend(["logistic", "svm", "tree", "k nearest"], bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0.)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    draw_three()
    # test_and_compare(rearrange_data(x_test), rearrange_data(x_train), y_test, y_train)
