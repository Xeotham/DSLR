#!../.venv/bin/python
from sys import argv, path
path.append("..")
path.append(".")
from numpy import ndarray, vectorize, array, zeros, argmax
from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from dslr_lib.errors import print_error
from dslr_lib.maths import normalize
from dslr_lib.regressions import predict_proba, houses_id, id_houses, houses_colors
from regression.logreg_train import prepare_dataset

def load_parameters() -> tuple[ndarray, ndarray, ndarray, ndarray, list]:
    """
    Loads the trained model parameters (weights) for each Hogwarts house from a CSV file.

    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray]:
            A tuple containing the weights for each house:
            (ravenclaw_weights, slytherin_weights, gryffindor_weights, hufflepuff_weights).
    """
    df = read_csv("../datasets/weights.csv")
    features = read_csv("../datasets/features.csv", header=None).values.flatten().tolist()
    return (
        array([df["ravenclaw"].values]).T,
        array([df["slytherin"].values]).T,
        array([df["gryffindor"].values]).T,
        array([df["hufflepuff"].values]).T,
        features
    )

def prepare_predictions(test_path: str, features: list) -> ndarray:
    """
    Prepares the test dataset for prediction by normalizing it using the training dataset statistics.

    Args:
        test_path (str): Path to the test dataset CSV file.
        features (list): List of feature names.

    Returns:
        ndarray: Normalized feature matrix for the test dataset.
    """
    # Load and prepare the test dataset
    test_df: DataFrame = read_csv(test_path, header=0).drop("Index", axis=1)
    _, test_x = prepare_dataset(test_df, fill=True, features=features)

    # Load and prepare the training dataset for normalization statistics
    train_df: DataFrame = read_csv("../datasets/dataset_train.csv", header=0).drop("Index", axis=1)
    _, train_x = prepare_dataset(train_df, features=features)

    # Normalize the test dataset using the training dataset statistics
    return normalize(test_x, train_x)

def generate_predictions(
    matrix_x: ndarray,
    matrix_t: tuple[ndarray, ndarray, ndarray, ndarray],
) -> ndarray:
    """
    Generates predictions for each Hogwarts house using the trained model weights.

    Args:
        matrix_x (ndarray): Normalized feature matrix for prediction.
        matrix_t (tuple[ndarray, ndarray, ndarray, ndarray]): Tuple of weights for each house.

    Returns:
        ndarray: Predicted house indices for each sample in the test dataset.
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
    Main function to load model parameters, prepare the test dataset, generate predictions,
    and save the results to a CSV file.

    Steps:
        1. Load the trained model parameters.
        2. Prepare the test dataset for prediction.
        3. Generate predictions for the test dataset.
        4. Map predicted house indices to house names.
        5. Save the predictions to a CSV file.

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
        assert len(argv) == 2, "Please pass a test dataset CSV file as a parameter."

        # Load the trained model parameters
        matrix_t = load_parameters()
        features = matrix_t[-1]
        matrix_t = matrix_t[:-1]

        # Prepare the test dataset for prediction
        test_x = prepare_predictions(argv[1], features)

        # Generate predictions for the test dataset
        houses = generate_predictions(test_x, matrix_t)

        # Map predicted house indices to house names
        house_map = vectorize(lambda x: id_houses[x])
        houses_csv = DataFrame(
            data=house_map(houses),
            columns=["Hogwarts House"],
        )

        # Save the predictions to a CSV file
        project_path = str(__file__)
        project_path = project_path[:-len("/logreg_predict.py")]
        path = "../datasets/houses.csv"
        if not path.startswith('/'):
            path = "/" + path
        path = project_path + path
        houses_csv.to_csv(path, index_label="Index")

    except AssertionError as e:
        print_error(str(e))
    except FileNotFoundError:
        print_error("FileNotFoundError: Provided file not found.")
    except PermissionError:
        print_error("PermissionError: Permission denied on provided file.")
    except EmptyDataError:
        print_error("EmptyDataError: Provided dataset is empty.")
    except KeyError as err:
        print_error(f"KeyError: {err} is not in the required file.")
    except KeyboardInterrupt:
        print_error("KeyboardInterrupt: Program execution interrupted by user.")
    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
