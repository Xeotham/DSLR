import numpy as np
from numpy import ndarray, exp


def sigmoid(
    z: ndarray
) -> ndarray:
    return 1 / (1 + exp(-z))


def calculate_gradient(
    matrix_x: ndarray,
    matrix_y: ndarray,
    matrix_t: ndarray,
) -> ndarray:
    """

    Args:
        matrix_x: The current set of features
        matrix_y: The vector of results
        matrix_t: The current set of parameters

    Returns:

    """
    m = matrix_y.size
    return (matrix_x.T @ (sigmoid(matrix_x @ matrix_t) - matrix_y)) / m


def gradient_descent(
    matrix_x: ndarray,
    matrix_y: ndarray,
    alpha: float =0.1,
    max_iter: int=100,
    tol: float=1e-7,
) -> ndarray:
    """

    Args:
        matrix_x: The current set of features
        matrix_y: The vector of results
        alpha:    The learning rate
        max_iter: The maximum number of iterations
        tol:    The tolerance rate

    Returns:
        matrix_t: A set of parameters
    """
    matrix_xb = np.c_[np.ones((matrix_x.shape[0], 1)), matrix_x]
    matrix_t = np.zeros((matrix_xb.shape[1], 1))

    for i in range(max_iter):
        gradient = calculate_gradient(matrix_xb, matrix_y, matrix_t)
        matrix_t -= alpha * gradient
        if np.linalg.norm(gradient) < tol:
            break
    return matrix_t


def predict_proba(
    matrix_x: ndarray,
    matrix_t: ndarray,
) -> float:
    matrix_xb = np.c_[np.ones((matrix_x.shape[0], 1)), matrix_x]
    return sigmoid(matrix_xb @ matrix_t)


def predict(
    matrix_x: ndarray,
    matrix_t: ndarray,
    threshold: float = 0.5
) -> int:
    return predict_proba(matrix_x, matrix_t) >= threshold

# pip install scikit-learn

# from sklearn.datasets import load_breast_cancer
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# x, y = load_breast_cancer(return_X_y=True)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#
# scaler = StandardScaler()
#
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)
#
# y_train.resize((y_train.shape[0], 1))
# thetas = gradient_descent(x_train_scaled, y_train)
#
# y_pred_train = predict(x_train_scaled, thetas)
# y_pred_test = predict(x_test_scaled, thetas)
#
# train_acc = accuracy_score(y_train, y_pred_train)
# test_acc = accuracy_score(y_test, y_pred_test)
#
# print(train_acc)
# print(test_acc)