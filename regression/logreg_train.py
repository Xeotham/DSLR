#!../.venv/bin/python
from sys import argv, path
path.append("..")
path.append(".")
path.append("../visualization")

from numpy import ndarray, array, zeros, argmax
from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError, ParserError
from pandas.api.types import is_numeric_dtype
from dslr_lib.maths import normalize
from dslr_lib.regressions import gradient_descent, predict_proba, prepare_dataset
from dslr_lib.errors import print_error

def regression_wrapper(
    matrix_y: ndarray,
    matrix_x: ndarray,
    houses: int
) -> ndarray:
    """
    Wrapper function for training a binary logistic regression model for a specific Hogwarts house.

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
    weights = gradient_descent(matrix_x, train_y, max_iter=1000, alpha=0.01)
    return weights.flatten()

def logreg_train(
    matrix_y: ndarray,
    matrix_x: ndarray,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Trains a multinomial logistic regression model for all Hogwarts houses.

    Args:
        matrix_y (ndarray): Target labels.
        matrix_x (ndarray): Feature matrix.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray]:
            Weights for each house (ravenclaw, slytherin, gryffindor, hufflepuff).
    """
    # Normalize the feature matrix
    norm_x = matrix_x.copy()
    norm_x = normalize(norm_x, matrix_x)

    # Train a model for each house
    t_ravenclaw = regression_wrapper(matrix_y, norm_x, 0)
    t_slytherin = regression_wrapper(matrix_y, norm_x, 1)
    t_gryffindor = regression_wrapper(matrix_y, norm_x, 2)
    t_hufflepuff = regression_wrapper(matrix_y, norm_x, 3)

    return t_ravenclaw, t_slytherin, t_gryffindor, t_hufflepuff

def generate_predictions(
    matrix_x: ndarray,
    matrix_t: tuple[ndarray, ndarray, ndarray, ndarray],
) -> ndarray:
    """
    Generates predictions for each Hogwarts house using the trained model weights.

    Args:
        matrix_x (ndarray): Feature matrix for prediction.
        matrix_t (tuple[ndarray, ndarray, ndarray, ndarray]): Tuple of weights for each house.

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

def main() -> None:
    """
    Main function to load the dataset, train a logistic regression model, and save the weights.

    Steps:
        1. Load the dataset.
        2. Prepare the feature and target matrices.
        3. Train the logistic regression model.
        4. Save the model weights to a CSV file.

    Raises:
        AssertionError: If the script is not called with the correct number of arguments.
        FileNotFoundError: If the dataset file is not found.
        PermissionError: If there is no permission to access the file.
        EmptyDataError: If the dataset is empty.
        KeyError: If a required column is missing from the dataset.
        KeyboardInterrupt: If the user interrupts the program execution.
        Exception: For any other unexpected errors.
    """
    try:
        # Ensure the script is called with the correct number of arguments
        assert len(argv) == 2, "Please pass a dataset CSV file as a parameter."

        # Load and prepare the dataset
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1)
        matrix_y, matrix_x = prepare_dataset(df, features=["Herbology", "Defense Against the Dark Arts"])
        matrix_y.resize((matrix_y.shape[0], 1))

        assert matrix_x.shape[0] != 0, "The dataset format isn't right."

        # Train the logistic regression model
        thetas_weights = logreg_train(matrix_y, matrix_x)

        features_csv = DataFrame(
            data=array(["Herbology", "Defense Against the Dark Arts"]),
        )
        # Save the model weights to a CSV file
        thetas_csv = DataFrame(
            data=array(thetas_weights).T,
            dtype=float,
            columns=["ravenclaw", "slytherin", "gryffindor", "hufflepuff"]
        )
        project_path = str(__file__)
        project_path = project_path[:-len("/logreg_train.py")]
        weights_path = "../datasets/weights.csv"
        features_path = "../datasets/features.csv"
        if not weights_path.startswith('/'):
            weights_path = "/" + weights_path
        if not features_path.startswith('/'):
            features_path = "/" + features_path
        weights_path = project_path + weights_path
        features_path = project_path + features_path
        thetas_csv.to_csv(weights_path, index=False)
        features_csv.to_csv(features_path, index=False, header=False)

    except AssertionError as err:
        print_error(f"Error: {err}")
    except FileNotFoundError:
        print_error("FileNotFoundError: Provided file not found.")
    except PermissionError:
        print_error("PermissionError: Permission denied on provided file.")
    except EmptyDataError:
        print_error("EmptyDataError: Provided dataset is empty.")
    except ParserError:
            print_error("ParserError: Impossible to parse the dataset.")
    except KeyError as err:
        print_error(f"KeyError: {err} is not in the required file.")
    except KeyboardInterrupt:
        print_error("KeyboardInterrupt: Program execution interrupted by user.")
    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
