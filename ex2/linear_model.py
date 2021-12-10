import numpy as np
import sklearn as skl
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotnine import *
""" X matrix after removing un wanted columns and *without an intercept 1*
unwanted labels including "id", "date", "lat", "long". the "price label was 
 removed as its our response vector, and the zip code as it doesnt have a 
 "Total order" and will be  used as the dummy variable"""
MATRIX_X: None
""" Y vector of the response we want to predict, a.k.a the price"""
RESPONSE_VEC_Y: None
""" the X matrix but *with an intercept 1, and one hot encoding with the dummy variable "zipcode" """
MATRIX_X_DUMMY = None
"""test - training size ration"""
SPLIT_RATION = 0.25



def fit_linear_regression(matrix_x, response_vec_y):
    """
    compute coefficients vector ‘w' and X's singular values
    :param matrix_x: m x d design matrix
    :param response_vec_y: m x 1 response vector
    :return: coefficients vector ‘w' and X's singular values
    """
    u, sig, vt = np.linalg.svd(matrix_x)  # x = USV.T, the singular values are the cols of sigma
    x_pseudo_inverse = np.linalg.pinv(matrix_x)
    w_hat = np.dot(x_pseudo_inverse, response_vec_y)  # w' = X.dagger * y
    return w_hat, sig


def predict(matrix_x, coefficients_vec_w):
    """
    compute the prediction rule vector i.e y_hat = Xw
    :param matrix_x: m x d design matrix
    :param coefficients_vec_w: m x 1 coefficients_vec_w
    :return: the prediction rule vector i.e y_hat = Xw
    """
    prediction = np.dot(matrix_x, coefficients_vec_w)
    j = 13
    return prediction


def mse(response_vec_y, prediction_vec_y_hat):
    """
    return the MSE- mean squared error  between the response vec and
    the prediction vec
    :param response_vec_y: the 'real' response vector
    :param prediction_vec_y_hat: the prediction vector the model learned
    :return: the MSE value
    """
    mse = skl.metrics.mean_squared_error(response_vec_y, prediction_vec_y_hat)
    return mse


def load_data(csv_path):
    """
    load the fataset and preforms the needed preprocessing in order to
    get a valid design matrix"
    :param csv_path: csv path with data to pre process
    :return: a single pandas 'DataFrame'
    """
    data_frame = pd.read_csv(csv_path)
    data_frame = data_frame.dropna()  # remove all rows with null values
    data_frame = data_frame[(data_frame.price > 0) & (data_frame.bedrooms > 0) & (data_frame.bathrooms > 0) &
                    (data_frame.sqft_living > 0) & (data_frame.sqft_lot > 0) & (data_frame.floors > 0) &
                    (data_frame.grade > 0)]  # filtering the none valid rows

    data_frame = data_frame.drop(["id", "date", "lat", "long"], axis=1)  # remove the unwanted cols

    data_frame.insert(0, "intercept", 1, True)  # adding the intercept for future calculations
    response_y = data_frame.price  # take a spesific col from the data frame
    matrix_x = data_frame.drop(["price", "zipcode"], axis=1)
    # using one hot encoding on 'zipcode' col
    x_with_dummy = pd.get_dummies(data_frame, columns=["zipcode"], drop_first=True)
    # using the drop_first cmd in order to  remove the zipcode clo and avoid the "Dummy Variable Trap"
    x_with_dummy = x_with_dummy.drop("price", axis=1)
    global MATRIX_X, RESPONSE_VEC_Y, MATRIX_X_DUMMY
    MATRIX_X = matrix_x.drop("intercept", axis=1)
    RESPONSE_VEC_Y, MATRIX_X_DUMMY = response_y, x_with_dummy # saving the pre proses data frame into a global vars

    # MATRIX_X.to_csv("MATRIX_X.csv")
    # x_with_dummy.to_csv("meme.csv")
    return x_with_dummy, response_y


def plot_singular_values(singular_val_collection):
    """
    create a Scree plot graph from a given singular values array
    :param singular_val_collection: ingular values array
    :return: void
    """
    sorted_arr = np.sort(singular_val_collection)
    sorted_desc_collection = sorted_arr[::-1]

    num_of_values = list(range(len(singular_val_collection)))

    plt.plot(num_of_values, sorted_desc_collection)
    plt.title("Scree Plot")
    plt.xlabel({"x - number of values"})
    plt.ylabel({"y - singular values value (in mil)"})
    plt.savefig("singular value ")
    plt.show()

def putting_it_all_together_1():
    """
    loads the data set, performs the pre-processing and plots the singular values
    array plot
    :return: void
    """
    # remark: to make clearer graph, the 'one hot encoding' doesnt effect the singular
    # values that much so will send the global X_MATRIX without the dummies
    matrix_x_with_dummy, response_y = load_data("kc_house_data.csv")
    singular_value_arr = np.linalg.svd(MATRIX_X, compute_uv=False)
    plot_singular_values(singular_value_arr)



def feature_evaluation(x, y):
    """
    calclation and ploting the pearson correlation between each feature and the price
    :param x: the data frame with all the non-categorical feature
    :param y: the responce vector represent the price
    :return: creat the scatter graph for each feature vector (col of x) and the response vec (y vec)
    """

    for col in x:
        # pearson correlation calculation
        cov_matrix = np.cov(x[col], y)
        pearson_numerator = cov_matrix[0][1]  # the secondary diagonal has the needed info to be the numerator for the 'Pearson Coleration'
        pearson_denominator = np.std(x[col]) * np.std(y)
        pearson_correlation = pearson_numerator / pearson_denominator

        # plot and save the result graphs
        plt.scatter(x[col], y, color='purple', alpha=0.6)
        plt.title("Pearson correlation between %s and the Price (response) \n" % col
                  + r"Correlation value $\rho=%0.4f$" % pearson_correlation)
        plt.xlabel("Features %s " % col)
        plt.ylabel("Price (in mils)")
        plt.savefig("Pearson correlation  " + str(col))
        plt.show()

def putting_it_all_together_2():
    """
    train and test the model co respond to the MSE division
    :return:
    """

    mse_array = []
    x_with_dummy, y_response = load_data("kc_house_data.csv")
    x_train, x_test, y_train, y_test = train_test_split(x_with_dummy, y_response, test_size=SPLIT_RATION)
    # the test and train steps
    for p in range(1, 40):
        print(p)
        cur_num_of_rows = max(int((len(x_train) * p )/100), 1)
        w_hat, sigma = fit_linear_regression(x_train[:cur_num_of_rows], y_train[:cur_num_of_rows])
        y_predict = predict(x_test, w_hat)
        mse_array.append(mse(y_test, y_predict))

    num_of_values = list(range(1, 40))
    plt.plot(num_of_values, mse_array)
    plt.title("MSE over the test set\n as a function p%")
    plt.xlabel("percentage")
    plt.ylabel("MSE (in bil)")
    plt.savefig("Training model")
    plt.show()
    # fig = go.Figure([go.Scatter(x=num_of_values, y=mse_array, name="traning model")],
    #                  layout=go.Layout(title="MSE over the test set\n as a function p%", xaxis="percentage",
    #                                   yaxis={"mse"}))
    # fig.show()


if __name__ == '__main__':
    print('hi')
    # x, y = load_data("kc_house_data.csv")
    # feature_evaluation(MATRIX_X, RESPONSE_VEC_Y)
    # putting_it_all_together_1()
    putting_it_all_together_2()





