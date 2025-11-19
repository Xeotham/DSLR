#!../.venv/bin/python
from sys import argv, path
path.append("..")
path.append(".")
path.append("../visualization")
from numpy import ndarray, vectorize, array, zeros, argmax
from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from pandas.api.types import is_numeric_dtype
from dslr_lib.errors import print_error
from dslr_lib.opti_bonus import cross_validation, logreg_train, FeaturesSelector, prepare_dataset
from dslr_lib.regressions import houses_id, id_houses, houses_colors

def features_name(features: ndarray, df: DataFrame) -> list:
    """
    Maps feature vectors to their corresponding column names in the DataFrame.

    Args:
        features (ndarray): A 2D array of feature vectors (each row is a feature).
        df (DataFrame): The DataFrame containing the original data and column names.

    Returns:
        list: A list of column names corresponding to each feature vector.
    """
    names = []
    def cmp_features(cmp_1, cmp_2):
        """
        Compares two feature vectors for equality.

        Args:
            cmp_1 (ndarray): First feature vector.
            cmp_2 (ndarray): Second feature vector.

        Returns:
            bool: True if all elements are equal, False otherwise.
        """
        for v_1, v_2 in zip(cmp_1, cmp_2):
            if v_1 != v_2:
                return False
        return True

    # For each feature vector, find the matching column in the DataFrame
    for feature in features.T:
        for name, values in zip(df.keys(), df.values.T):
            if cmp_features(feature, values):
                names.append(name)
                break
    return names

def main():
    """
    Main function to train a logistic regression model, select features, and save the weights.

    Steps:
        1. Load and prepare the dataset.
        2. Train the logistic regression model.
        3. Select the best features.
        4. Evaluate the model using cross-validation.
        5. Save the model weights to a CSV file.

    Raises:
        AssertionError: If the script is not called with the correct number of arguments.
        FileNotFoundError: If the provided dataset file does not exist.
        PermissionError: If there is no permission to access the file.
        EmptyDataError: If the dataset is empty.
        KeyError: If a required column is missing from the dataset.
        KeyboardInterrupt: If the user interrupts the program execution.
    """
    try:
        # Ensure the script is called with the correct number of arguments
        assert len(argv) == 2, "Please pass a \"dataset_train.csv\" as parameter"

        # Load and prepare the dataset
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1)
        matrix_y, matrix_x = prepare_dataset(df)
        matrix_y.resize((matrix_y.shape[0], 1))

        # Select the best features
        features_select = FeaturesSelector(matrix_x, matrix_y, 4, 0.98)
        matrix_x = features_select.find_best_features()

        # Evaluate the model using cross-validation
        p_score = cross_validation(matrix_x, matrix_y)
        print(f"Accuracy: {p_score}")

        # Train the model with the selected features
        thetas_weights = logreg_train(matrix_x, matrix_y)

        # Save the model weights to a CSV file
        thetas_csv = DataFrame(
            data=array(thetas_weights).T,
            dtype=float,
            columns=["ravenclaw", "slytherin", "gryffindor", "hufflepuff"]
        )
        project_path = str(__file__)
        project_path = project_path[:-len("/logreg_train_bonus.py")]
        path = "../datasets/weights.csv"
        if not path.startswith('/'):
            path = "/" + path
        path = project_path + path
        thetas_csv.to_csv(path, index=False)

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
