import sys
from typing import Any, Tuple, Optional
from numpy.ma.extras import hstack
sys.path.append('../')
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype
from multiprocessing import cpu_count, Queue, Manager, Pool
from numpy import (
    ndarray, array, mean, argmax,
    arange, concatenate, zeros
)
from numpy.random import shuffle as np_shuffle
from dslr_lib.regressions import gradient_descent, predict_proba, houses_id
from dslr_lib.maths import normalize
from dslr_lib.threads_bonus import threaded

def prepare_dataset(
    df: DataFrame,
    fill: bool = False
) -> Tuple[ndarray, ndarray]:
    """
    Prepares the dataset for machine learning by handling missing values and extracting features and labels.

    Args:
        df (DataFrame): Input DataFrame containing the dataset.
        fill (bool, optional): If True, fills missing values with the mean of their column. If False, drops rows with missing values. Defaults to False.

    Returns:
        Tuple[ndarray, ndarray]: A tuple containing the target labels (matrix_y) and feature matrix (matrix_x).
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

    # Extract target labels and features
    matrix_y = df[df.columns[0]]
    if "Hogwarts House" in df.columns:
        matrix_y = df["Hogwarts House"].map(houses_id)
    matrix_x = df.select_dtypes(include="number")
    return matrix_y.values, matrix_x.values

def regression_wrapper(
    matrix_y: ndarray,
    matrix_x: ndarray,
    houses: int
) -> ndarray:
    """
    Wrapper function for training a logistic regression model for a specific house.

    Args:
        matrix_y (ndarray): Target labels.
        matrix_x (ndarray): Feature matrix.
        houses (int): Index of the house to train the model for.

    Returns:
        ndarray: Flattened weights of the trained logistic regression model.
    """
    # Create binary labels for the specified house
    train_y = matrix_y.copy()
    train_y[matrix_y != houses] = 0
    train_y[matrix_y == houses] = 1

    # Train the logistic regression model
    weights = gradient_descent(matrix_x, train_y, max_iter=5000, alpha=0.01)
    return weights.flatten()

def logreg_train(
    matrix_y: ndarray,
    matrix_x: ndarray,
    multi_process: bool = False
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Trains a multinomial logistic regression model for all houses.

    Args:
        matrix_y (ndarray): Target labels.
        matrix_x (ndarray): Feature matrix.
        multi_process (bool, optional): If True, a calling process already use multiprocessing. Defaults to False.

    Returns:
        Tuple[ndarray, ndarray, ndarray, ndarray]: Weights for each house (ravenclaw, slytherin, gryffindor, hufflepuff).
    """
    # Normalize the feature matrix
    norm_x = normalize(matrix_x, matrix_x)

    if not multi_process:
        # Use multiprocessing to train models for each house
        with Pool(4) as pool:
            res = [
                pool.apply_async(regression_wrapper, (matrix_y, norm_x, i))
                for i in range(4)
            ]
            t_ravenclaw = res[0].get()
            t_slytherin = res[1].get()
            t_gryffindor = res[2].get()
            t_hufflepuff = res[3].get()
            pool.close()
            pool.join()
    else:
        # Train models sequentially
        t_ravenclaw = regression_wrapper(matrix_y, norm_x, 0)
        t_slytherin = regression_wrapper(matrix_y, norm_x, 1)
        t_gryffindor = regression_wrapper(matrix_y, norm_x, 2)
        t_hufflepuff = regression_wrapper(matrix_y, norm_x, 3)

    return t_ravenclaw, t_slytherin, t_gryffindor, t_hufflepuff

def generate_predictions(
    matrix_x: ndarray,
    matrix_t: Tuple[ndarray, ...],
) -> ndarray:
    """
    Generates predictions for each house using the trained model weights.

    Args:
        matrix_x (ndarray): Feature matrix.
        matrix_t (Tuple[ndarray, ...]): Tuple of weights for each house.

    Returns:
        ndarray: Predicted house indices for each sample.
    """
    # Initialize predictions matrix
    predictions = zeros((matrix_x.shape[0], 4))

    # Generate probabilities for each house
    for i in range(predictions.shape[1]):
        predicted = predict_proba(matrix_x, matrix_t[i])
        predictions[:, i] = predicted.ravel()

    # Select the house with the highest probability for each sample
    chosen_houses = zeros((matrix_x.shape[0], 1))
    for i in range(chosen_houses.shape[0]):
        chosen_houses[i][0] = argmax(predictions[i])

    return chosen_houses

def cross_validation(
    X: ndarray,
    y: ndarray,
    k: int = 5,
    multi_process: bool = False,
) -> float:
    """
    Performs k-fold cross-validation for a multinomial logistic regression model.

    Args:
        X (ndarray): Input feature matrix of shape (n_samples, n_features).
        y (ndarray): Target labels of shape (n_samples,).
        k (int, optional): Number of folds for cross-validation. Defaults to 5.
        multi_process (bool, optional): If True, a calling process use multiprocess. Defaults to False.

    Returns:
        float: Average accuracy of the model across all folds.
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = arange(n_samples)
    np_shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    accuracies = []
    threads = []

    def verif(iter_number: int) -> float:
        """
        Evaluates the model on a single fold.

        Args:
            iter_number (int): Index of the current fold.

        Returns:
            float: Accuracy of the model on the validation fold.
        """
        val_start = iter_number * fold_size
        val_end = (iter_number + 1) * fold_size
        X_val = X_shuffled[val_start:val_end]
        y_val = y_shuffled[val_start:val_end]
        X_train = concatenate([X_shuffled[:val_start], X_shuffled[val_end:]])
        y_train = concatenate([y_shuffled[:val_start], y_shuffled[val_end:]])

        # Train the model and generate predictions
        weights = logreg_train(y_train, X_train, multi_process=True)
        y_pred = generate_predictions(X_val, weights)
        accuracy = mean(y_pred == y_val)
        return accuracy

    @threaded
    def thread_verif(iter_number: int, queue: Queue) -> None:
        """
        Threaded wrapper for evaluating the model on a single fold.

        Args:
            iter_number (int): Index of the current fold.
            queue (Queue): Queue to store the accuracy result.
        """
        queue.put(verif(iter_number))

    if not multi_process:
        # Use multiprocessing for cross-validation
        with Manager() as manager:
            q = manager.Queue()
            for i in range(k):
                threads.append(thread_verif(i, q))
            for thread in threads:
                thread.join()
                accuracies.append(q.get())
    else:
        # Perform cross-validation sequentially
        for i in range(k):
            accuracies.append(verif(i))

    return mean(accuracies)

class FeaturesSelector:
    """
    A class for selecting the best features based on cross-validation accuracy.
    """

    def __init__(
        self,
        matrix_x: ndarray,
        matrix_y: ndarray,
        f_number: int,
        acc_tol: float
    ) -> None:
        """
        Initializes the FeaturesSelector with the feature matrix, target labels, number of features, and accuracy tolerance.

        Args:
            matrix_x (ndarray): Feature matrix.
            matrix_y (ndarray): Target labels.
            f_number (int): Number of features to select.
            acc_tol (float): Minimum accuracy tolerance for feature selection.
        """
        self.matrix_X = matrix_x
        self.matrix_y = matrix_y
        self.f_number = f_number
        self.acc_tol = acc_tol

    def __test_features(
        self,
        main_feature: ndarray,
        f_test: Optional[ndarray],
        is_finished: Any
    ) -> Tuple[float, Optional[ndarray]]:
        """
        Recursively tests feature combinations to find the best set based on cross-validation accuracy.

        Args:
            main_feature (ndarray): Matrix of features to test.
            f_test (Optional[ndarray]): Current set of features being tested.
            is_finished (Any): Flag to indicate if the search should stop early.

        Returns:
            Tuple[float, Optional[ndarray]]: Best accuracy and corresponding feature set.
        """
        best_score = float("-inf")
        best_features = None

        if self.f_number == 1 and f_test.shape[1] == 1:
            return cross_validation(f_test, self.matrix_y, multi_process=True), f_test

        for i in range(0, main_feature.shape[1]):
            if is_finished.value:
                return best_score, best_features

            if f_test is None:
                actual_features = array([main_feature[:, i]]).T
            else:
                actual_features = array(hstack((f_test, main_feature[:, i].reshape(main_feature.shape[0], 1))))

            if actual_features.shape[1] < self.f_number and i + 1 < main_feature.shape[1]:
                tmp_score, tmp_features = self.__test_features(main_feature[:, i + 1:], actual_features, is_finished)
                if tmp_score > best_score:
                    best_score = tmp_score
                    best_features = tmp_features
                continue
            elif i + 1 >= main_feature.shape[1]:
                return best_score, best_features

            actual_score = cross_validation(actual_features, self.matrix_y, multi_process=True)
            if actual_score > best_score:
                if actual_score >= self.acc_tol:
                    is_finished.value = True
                best_score = actual_score
                best_features = actual_features

        return best_score, best_features

    def find_best_features(self) -> ndarray:
        """
        Finds the best feature set using multiprocessing.

        Returns:
            ndarray: Best feature set based on cross-validation accuracy.
        """
        @threaded
        def test_features_wrapper(
            main_feature: ndarray,
            f_test: ndarray,
            queue: Queue,
            is_finished: Any
        ) -> None:
            """
            Threaded wrapper for testing feature combinations.

            Args:
                main_feature (ndarray): Matrix of features to test.
                f_test (ndarray): Current set of features being tested.
                queue (Queue): Queue to store the result.
                is_finished (Any): Flag to indicate if the search should stop early.
            """
            queue.put(self.__test_features(main_feature, f_test, is_finished))

        thread_iter = (self.matrix_X.shape[1] / cpu_count()).__ceil__()
        threads = []
        result = []
        best_score = float("-inf")
        best_features = None

        if self.f_number == self.matrix_X.shape[1]:
            return self.matrix_X.copy()

        with Manager() as manager:
            q = manager.Queue()
            finished = manager.Value('b', False)
            for i in range(0, thread_iter):
                for j in range(cpu_count()):
                    if (i * thread_iter) + j > self.matrix_X.shape[1] - self.f_number:
                        break
                    threads.append(
                        test_features_wrapper(
                            self.matrix_X[:, (i * thread_iter) + j + 1:],
                            array([self.matrix_X[:, (i * thread_iter) + j]]).T,
                            q,
                            finished
                        )
                    )
                for thread in threads:
                    thread.join()
                    result.append(q.get())

        for score, features in result:
            if score > best_score:
                best_score = score
                best_features = features

        return best_features
