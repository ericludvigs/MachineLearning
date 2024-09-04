import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def FrankeFunction(x: ArrayLike,y: ArrayLike) -> ArrayLike:
    """Evaluates the Franke function, provided by project text
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def make_feature_matrix(x: ArrayLike, P: int) -> ArrayLike:
    """Creates a feature matrix X for some input vector x

    X has dimension n*P with n=len(x) and P features
    
    First row is excluded, leaving out the intercept (making scaling easy)
    """
    X: ArrayLike = np.zeros((len(x), P-1))
    # leave out intercept by excluding the first row of X
    for exponent in range(1,P):
        X[:,exponent-1] = x[:]**exponent

    return X

def scale_X_and_y(X: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Scale by subtracting mean from each column of a feature matrix X and output data vector y
    """
    y_mean = np.mean(y)
    # assume X is a n*P feature matrix
    X_mean = np.mean(X,axis=0)

    X = X - X_mean
    y = y - y_mean
    return X, y

def linear_regression_model_OLS(X: ArrayLike, y: ArrayLike) -> ArrayLike:
    """ Regression model using Ordinary Least Squares with matrix inversion to calculate a model vector beta (dimension P)
    """
    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    # from https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter1.html#linear-regression-basic-elements

    return beta

def linear_regression_model_Ridge(X: ArrayLike, y: ArrayLike, lambda_value: float) -> ArrayLike:
    """ Regression model based on Ordinary Least Squares, but with hyperparameter lambda added resulting in Ridge regression

    Uses matrix inversion and returns a model vector beta (dimension P)
    """
    beta = np.linalg.pinv( X.T.dot(X) + lambda_value * np.identity(X.shape[1]) ).dot(X.T).dot(y)

    return beta

def linear_prediction(X: ArrayLike, beta: ArrayLike) -> ArrayLike:
    """ Perform simple linear prediction via matrix product of feature matrix X and model vector beta

    Gives output vector y of dimension n
    """
    y_tilde = X @ beta

    return y_tilde   

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')


    ### Parameters ###
    np.random.seed(4242)

    max_polynomial_degree = 5


    ### Make data ###
    polynomial_degrees = np.arange(1, max_polynomial_degree+1, 1)
    print(polynomial_degrees)
    x_1d = np.arange(0, 1, 0.05)
    y_1d = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x_1d,y_1d)

    print(f"x grid shape: {x.shape}")
    print(f"y grid shape: {y.shape}")
    # simple len works with these grids
    n = len(x)

    z = FrankeFunction(x, y)

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

