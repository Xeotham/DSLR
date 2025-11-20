from numpy import ndarray, exp, c_, ones, zeros, empty_like
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype
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

def prepare_dataset(
    df: DataFrame,
    fill: bool = False,
    features: list = None,
) -> tuple[ndarray, ndarray]:
    """
    Prepares the dataset for logistic regression by handling missing values and extracting features and labels.
    Only the "Herbology" and "Defense Against the Dark Arts" features are selected for this analysis.

    Args:
        df (DataFrame): Input DataFrame containing the dataset.
        fill (bool, optional): If True, fills missing values with the mean of their column.
                              If False, drops rows with missing values. Defaults to False.

    Returns:
        tuple[ndarray, ndarray]: A tuple containing the target labels (matrix_y) and feature matrix (matrix_x).
    """
    if fill:
        # Fill missing values with the mean of their column
        for i in range(0, len(df.columns)):
            if not is_numeric_dtype(df[df.columns[i]]):
                continue
            df[df.columns[i]] = df[df.columns[i]].fillna(df[df.columns[i]].mean())
    else:
        # Drop rows with missing values
        df.dropna(inplace=True)

    # Extract target labels
    matrix_y = df[df.columns[0]]
    if "Hogwarts House" in df.columns:
        matrix_y = df["Hogwarts House"].map(houses_id)

    # Select only the "Herbology" and "Defense Against the Dark Arts" features
    matrix_x = df.select_dtypes(include="number")
    if features:
        matrix_x = matrix_x[features]
    return matrix_y.values, matrix_x.values

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
