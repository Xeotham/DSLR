import sys
sys.path.append('../')

from numpy import ndarray, argmin, abs, unique, stack, sum, mean, argmax, zeros_like, arange, concatenate, zeros
from numpy.random import shuffle
from dslr_lib.regressions import gradient_descent, predict_proba
from dslr_lib.maths import normalize

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
    X,
    y,
    k=5,
):
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

