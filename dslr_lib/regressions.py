from numpy import ndarray, exp, c_, ones, zeros, empty_like
from numpy.linalg import norm

houses_colors = {
    "Ravenclaw": "c",
    "Gryffindor": "r",
    "Slytherin": "g",
    "Hufflepuff": "y"
}

houses_id = {
    "Ravenclaw": 0,
    "Gryffindor": 1,
    "Slytherin": 2,
    "Hufflepuff": 3
}

id_houses = {
    0: "Ravenclaw",
    1: "Gryffindor",
    2: "Slytherin",
    3: "Hufflepuff",
}

def sigmoid(
    z: ndarray
) -> ndarray:
    """
    Applies a sigmoid function h to each value of the matrix z,
    where h(x) = 1 / (1 + e^(x))

    Args:
        z: A matrix

    Returns:
        ndarray: A new matrix where each element v has been mapped to h(v)
    """
    out = empty_like(z)
    pos_mask = (z >= 0)
    out[pos_mask] = 1 / (1 + exp(-z[pos_mask]))
    neg_mask = ~pos_mask
    exp_z = exp(z[neg_mask])
    out[neg_mask] = exp_z / (1 + exp_z)

    return out


def calculate_gradient(
    matrix_x: ndarray,
    matrix_y: ndarray,
    matrix_t: ndarray,
) -> ndarray:
    """
    Calculates the gradients in order to then minimizes the loss function
    of our logistic regression

    Args:
        matrix_x: The current set of features
        matrix_y: The vector of results
        matrix_t: The current set of parameters

    Returns:
        ndarray: A matrix of gradient that will be used to minimize the cost
            function
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
    Minimizes the log loss function in order using the gradient
    descent algorithm
    The function effectively executes a logistic regression using
    the features (x) and the result (y) to classify

    Args:
        matrix_x: The current set of features
        matrix_y: The vector of results
        alpha:    The learning rate
        max_iter: The maximum number of iterations
        tol:    The tolerance rate

    Returns:
        matrix_t: A set of parameters to be able to classify and do predictions
    """
    matrix_xb = c_[ones((matrix_x.shape[0], 1)), matrix_x]
    matrix_t = zeros((matrix_xb.shape[1], matrix_y.shape[1]))

    for i in range(max_iter):
        gradient = calculate_gradient(matrix_xb, matrix_y, matrix_t)
        matrix_t -= alpha * gradient
        if norm(gradient) < tol:
            break
    return matrix_t


def predict_proba(
    matrix_x: ndarray,
    matrix_t: ndarray,
) -> ndarray:
    """
    Uses a matrix of weight parameters and an input vector of features
    to predict the chance of each values of being what we're classifying

    Args:
        matrix_x: input vector of features
        matrix_t: matrix of weight parameters

    Returns:
        ndarray: A matrix of probability
    """


    matrix_xb = c_[ones((matrix_x.shape[0], 1)), matrix_x]
    return sigmoid(matrix_xb @ matrix_t)


def predict(
    matrix_x: ndarray,
    matrix_t: ndarray,
    threshold: float = 0.5
) -> ndarray[bool]:
    """
    A wrapper around predict_proba() that checks if the chances of the value
    being 'true' is higher than the threshold

    Args:
        matrix_x: input vector of features
        matrix_t: matrix of weight parameters
        threshold: the least probability threshold required to return true

    Returns:
        ndarray: A matrix of boolean
    """
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