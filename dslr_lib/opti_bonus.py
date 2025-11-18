import sys
from multiprocessing.managers import ValueProxy
from typing import Any

import numpy as np
from numpy.ma.extras import hstack

sys.path.append('../')

from multiprocessing import cpu_count, Value, Queue, Lock, Manager
from numpy import ndarray, array, argmin, abs, unique, stack, sum, mean, argmax, zeros_like, arange, concatenate, zeros, vstack
from numpy.random import shuffle
from dslr_lib.regressions import gradient_descent, predict_proba
from dslr_lib.maths import normalize
from dslr_lib.threads_bonus import threaded


def regression_wrapper(
    matrix_y: ndarray[int],
    matrix_x: ndarray[float],
    houses: int
) -> ndarray[float]:
    train_y = matrix_y.copy()
    train_y[matrix_y != houses] = 0
    train_y[matrix_y == houses] = 1

    weights = gradient_descent(matrix_x, train_y, max_iter=5000, alpha=0.01)
    return weights.flatten()

def logreg_train(
    matrix_y: ndarray[int],
    matrix_x: ndarray[float],
) -> tuple[ndarray[float], ndarray[float], ndarray[float], ndarray[float]]:
    norm_x = matrix_x.copy()
    norm_x = normalize(norm_x, matrix_x)
    norm_x = norm_x
    matrix_y = matrix_y
    t_ravenclaw = regression_wrapper(matrix_y, norm_x, 0)
    t_slytherin = regression_wrapper(matrix_y, norm_x, 1)
    t_gryffindor = regression_wrapper(matrix_y, norm_x, 2)
    t_hufflepuff = regression_wrapper(matrix_y, norm_x, 3)

    return t_ravenclaw, t_slytherin, t_gryffindor, t_hufflepuff

# TODO: Place on dslr_libs
def generate_predictions(
    matrix_x: ndarray,
    matrix_t: tuple,
) -> ndarray:
    predictions = zeros((matrix_x.shape[0], 4))
    for i in range(predictions.shape[1]):
        predicted = predict_proba(matrix_x, matrix_t[i])
        predictions[:, i] = predicted.ravel()
    chosen_houses = zeros((matrix_x.shape[0], 1))
    for i in range(chosen_houses.shape[0]):
        chosen_houses[i][0] = argmax(predictions[i])
    return chosen_houses

def cross_validation(
    X: ndarray,
    y: ndarray,
    k: int = 5
) -> float:
    """
    Perform k-fold cross-validation for a multinomial logistic regression model.

    Args:
        X (ndarray): Input feature matrix of shape (n_samples, n_features).
        y (ndarray): Target labels of shape (n_samples,).
        k (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
        float: Average accuracy of the model across all folds.
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = arange(n_samples)
    shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    accuracies = []
    for i in range(k):
        # Split into training and validation sets
        val_start = i * fold_size
        val_end = (i + 1) * fold_size
        X_val = X_shuffled[val_start:val_end]
        y_val = y_shuffled[val_start:val_end]
        X_train = concatenate([X_shuffled[:val_start], X_shuffled[val_end:]])
        y_train = concatenate([y_shuffled[:val_start], y_shuffled[val_end:]])
        # Train multinomial logistic regression
        weights = logreg_train(y_train, X_train)
        y_pred = generate_predictions(X_val, weights)
        accuracy = mean(y_pred == y_val)
        accuracies.append(accuracy)
    return mean(accuracies)

class FeaturesSelector:
    matrix_X: ndarray
    matrix_y: ndarray
    f_number: int
    acc_tol: float

    def __init__(self, matrix_x, matrix_y, f_number, acc_tol):
        self.matrix_X = matrix_x
        self.matrix_y = matrix_y
        self.f_number = f_number
        self.acc_tol = acc_tol


    def __test_features(self, main_feature: ndarray, f_test: ndarray, is_finished):
        best_score = float("-inf")
        best_features = None

        for i in range(0, main_feature.shape[1]):
            if is_finished.get():
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

            actual_score = cross_validation(actual_features, self.matrix_y)
            if actual_score > best_score:
                if actual_score >= self.acc_tol:
                    is_finished.set(True)
                best_score = actual_score
                best_features = actual_features
        return best_score, best_features

    def find_best_features(self):
        # TODO: Multi thread avec NPROC
        @threaded
        def test_features_wrapper(main_feature: ndarray, f_test: ndarray, queue: Queue, is_finished):
            queue.put(self.__test_features(main_feature, f_test, is_finished))

        thread_iter = (self.matrix_X.shape[1] / cpu_count()).__ceil__()
        threads = []
        result = []
        best_score = float("-inf")
        best_features = None

        with Manager() as manager:
            q = manager.Queue()
            finished = manager.Value('B', False)

            for i in range(0, thread_iter):
                for j in range(cpu_count()):
                    if (i * thread_iter) + j > self.matrix_X.shape[1] - 1:
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

# def test_features(
#     matrix_x: ndarray,
#     f_number: int,
#     current_number: int,
#     base_feature: ndarray,
#     *args
# ):
#     if current_number < f_number:
#         return test_features(matrix_x, f_number, base_feature, *args, )
#
#     return None
#
#     # test_x = matrix_x[:, 0].copy()
#
#
# def features_selector(
#     X: ndarray,
#     y: ndarray,
#     f_number: int = 3
# ):
#     test_features(X, f_number)
