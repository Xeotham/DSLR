#!../.venv/bin/python
from sys import path, argv
path.append("..")
path.append(".")
path.append("../visualization")
from numpy.ma.extras import hstack
from numpy import ndarray, array
from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from dslr_lib.maths import normalize
from dslr_lib.errors import print_error
from dslr_lib.opti_bonus import logreg_train
from dslr_lib.regressions import houses_colors, id_houses, prepare_dataset
from matplotlib.pyplot import plot, scatter, show

def plot_boundaries(
    matrix_y: ndarray,
    matrix_x: ndarray,
    matrix_t: ndarray,
    houses: int
) -> None:
    """
    Plots the decision boundaries for a logistic regression model for a specific house.

    Args:
        matrix_y (ndarray): Target labels.
        matrix_x (ndarray): Feature matrix.
        matrix_t (ndarray): Weights of the logistic regression model.
        houses (int): Index of the house for which to plot the boundary.
    """
    # Extract bias and weights
    b = matrix_t[-1]
    w = matrix_t[0:-1]

    # Calculate slope and intercept for the decision boundary
    m = w[0] / w[1]
    c = b / w[1]

    # Set plot boundaries
    xmin, xmax = matrix_x[:, 0].min() - 0.5, matrix_x[:, 0].max() + 0.5
    ymin, ymax = matrix_y.min() - 0.5, matrix_y.max() + 0.5

    # Generate points for the decision boundary line
    xd = array([xmin, xmax])
    yd = m * xd + c

    # Plot the decision boundary
    plot(xd, yd, ls='--', color=houses_colors[id_houses[houses]])

def main() -> None:
    """
    Main function to load the dataset, train a logistic regression model,
    and plot the decision boundaries and data points for each house.

    Steps:
        1. Load the dataset.
        2. Prepare the feature and target matrices.
        3. Train the logistic regression model.
        4. Plot the decision boundaries for each house.
        5. Plot the data points for each house.
        6. Display the plot.

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
        matrix_y, matrix_x = prepare_dataset(df, features=["Herbology", "Defense Against the Dark Arts"])
        matrix_y.resize((matrix_y.shape[0], 1))

        # Train the logistic regression model
        weights = logreg_train(matrix_y, matrix_x)

        # Normalize the feature matrix
        norm_x = normalize(matrix_x, matrix_x)

        # Plot the decision boundaries for each house
        plot_boundaries(matrix_y, norm_x, weights[0], 0)
        plot_boundaries(matrix_y, norm_x, weights[1], 1)
        plot_boundaries(matrix_y, norm_x, weights[2], 2)
        plot_boundaries(matrix_y, norm_x, weights[3], 3)

        # Combine target labels and features for plotting
        full_df = hstack((matrix_y, norm_x))

        # Separate data points by house
        separated_house = {
            id_houses[house]: full_df[full_df[:, 0] == house][:, 1:]
            for house in range(0, 4)
        }

        # Plot the data points for each house
        for house, h_df in separated_house.items():
            scatter(
                h_df[:, 0],
                h_df[:, 1],
                color=houses_colors[house],
                alpha=0.7,
            )

        # Display the plot
        show()

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
