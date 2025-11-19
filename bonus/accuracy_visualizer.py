#!../.venv/bin/python
from sys import path
path.append("..")
path.append(".")
from numpy import array, ndarray
from dslr_lib.opti_bonus import FeaturesSelector, cross_validation, prepare_dataset
from dslr_lib.errors import print_error
from dslr_lib.threads_bonus import threaded
from pandas import DataFrame, read_csv
from pandas.errors import EmptyDataError
from matplotlib.pyplot import bar, show, hist
from multiprocessing import Manager, Pool


def find_opti(matrix_x: ndarray, matrix_y: ndarray) -> None:
    """
    Finds the optimal feature set for each tolerance level and plots their cross-validation accuracies.

    Args:
        matrix_x (ndarray): Feature matrix (independent variables).
        matrix_y (ndarray): Target matrix (dependent variable).

    Plots:
        A bar chart of cross-validation accuracies for each tolerance level.
    """
    features_accuracy = []
    # Tolerance levels for feature selection
    accuracy_tol = [0.60, 0.98, 0.98, 0.98, 0.98, 0.98, 0.64, 0.64, 0.64, 0.64, 0.61, 0.25, 0.28]

    # For each tolerance level, select features and compute cross-validation accuracy
    for i, tol in enumerate(accuracy_tol):
        f_select = FeaturesSelector(matrix_x, matrix_y, i, tol)
        features_accuracy.append(cross_validation(f_select.find_best_features(), matrix_y))

    # Convert accuracies to DataFrame for plotting
    features_accuracy = DataFrame(
        data=array(features_accuracy).T,
        dtype=float,
    )

    # Plot the accuracies as a bar chart
    bar(
        x=range(len(features_accuracy)),
        height=array(features_accuracy),
        tick_label=[f"{i + 1}" for i in range(len(features_accuracy))]
    )
    show()

def hard_features(matrix_x: ndarray, matrix_y: ndarray, df: DataFrame) -> None:
    """
    Evaluates cross-validation accuracy for predefined feature sets and plots the results.

    Args:
        matrix_x (ndarray): Feature matrix (independent variables).
        matrix_y (ndarray): Target matrix (dependent variable).
        df (DataFrame): Original DataFrame containing all features.

    Plots:
        A bar chart of cross-validation accuracies for each predefined feature set.
    """
    features_accuracy = []
    threads = []

    # Predefined feature sets to evaluate
    features_list = [
        ["Defense Against the Dark Arts"],
        ['Herbology', 'Defense Against the Dark Arts'],
        ['Herbology', 'Defense Against the Dark Arts', 'Divination'],
        ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'History of Magic'],
        ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'History of Magic', 'Potions'],
        ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'History of Magic', 'Potions',
         'Care of Magical Creatures'],
        ['Astronomy', 'Herbology', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Care of Magical Creatures',
         'Charms'],
        ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes',
         'Care of Magical Creatures', 'Charms'],
        ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes',
         'History of Magic', 'Care of Magical Creatures', 'Charms'],
        ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes',
         'History of Magic', 'Potions', 'Care of Magical Creatures', 'Charms'],
        ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes',
         'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms'],
        ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
         'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms'],
        ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
         'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
         'Flying']
    ]

    for i, names in enumerate(features_list):
        features_accuracy.append(cross_validation(df[names].values, matrix_y))

    # Plot the accuracies as a bar chart
    bar(
        x=range(len(features_accuracy)),
        height=array(features_accuracy),
        tick_label=[f"{i+1}" for i in range(len(features_accuracy))]
    )
    show()

def main() -> None:
    """
    Main function to load the dataset, prepare features, and evaluate feature sets.

    Steps:
        1. Load the dataset.
        2. Prepare the feature and target matrices.
        3. Evaluate predefined feature sets or find optimal features.
        4. Plot the results.

    Raises:
        FileNotFoundError: If the dataset file is not found.
        PermissionError: If there is no permission to access the file.
        EmptyDataError: If the dataset is empty.
        KeyError: If a required column is missing from the dataset.
        KeyboardInterrupt: If the user interrupts the program execution.
        Exception: For any other unexpected errors.
    """
    try:
        # Load the dataset
        df: DataFrame = read_csv("../datasets/dataset_train.csv", header=0).drop("Index", axis=1)

        # Prepare the feature and target matrices
        matrix_y, matrix_x = prepare_dataset(df)
        matrix_y.resize((matrix_y.shape[0], 1))

        # Evaluate feature sets
        # find_opti(matrix_x, matrix_y)
        hard_features(matrix_x, matrix_y, df)

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

if __name__ == '__main__':
    main()