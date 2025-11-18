#!../.venv/bin/python

import sys
sys.path.insert(1, "..")
sys.path.insert(2, ".")
sys.path.insert(3, "../visualization")


from sys import argv
from numpy.ma.extras import hstack
from numpy import ndarray, array
from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from dslr_lib.maths import normalize
from dslr_lib.errors import print_error
from dslr_lib.opti_bonus import logreg_train
from matplotlib.pyplot import figure, plot, scatter, show, legend, xlim, ylim, fill_between
from logreg_train_bonus import prepare_dataset

# TODO: Place on dslr_libs
houses_colors = {
    "Ravenclaw": "c",
    "Gryffindor": "r",
    "Slytherin": "g",
    "Hufflepuff": "y"
}

# TODO: Place on dslr_libs
houses_id = {
    "Ravenclaw": 0,
    "Gryffindor": 1,
    "Slytherin": 2,
    "Hufflepuff": 3
}

# TODO: Place on dslr_libs
id_houses = {
    0: "Ravenclaw",
    1: "Gryffindor",
    2: "Slytherin",
    3: "Hufflepuff",
}

def plot_boundaries(
    matrix_y: ndarray[int],
    matrix_x: ndarray[float],
    matrix_t: ndarray[float],
    houses: int
):
    b = matrix_t[-1]
    w = matrix_t[0 : -1]
    m = w[0] / w[1]
    c = b / w[1]
    xmin, xmax = matrix_x[:, 0].min() - 0.5, matrix_x[:, 0].max() + 0.5
    ymin, ymax = matrix_y[:, 0].min() - 0.5, matrix_y[:, 0].max() + 0.5
    xd = array([xmin, xmax])
    yd = m * xd + c
    plot(xd, yd, ls='--', color=houses_colors[id_houses[houses]])


def main():
    try:
        df: DataFrame = read_csv("../datasets/dataset_train.csv", header=0).drop("Index", axis=1)
        matrix_y, matrix_x = prepare_dataset(df)
        matrix_y.resize((matrix_y.shape[0], 1))
        weights = logreg_train(matrix_y, matrix_x)
        norm_x = normalize(matrix_x, matrix_x)

        plot_boundaries(matrix_y, norm_x, weights[0], 0)
        plot_boundaries(matrix_y, norm_x, weights[1], 1)
        plot_boundaries(matrix_y, norm_x, weights[2], 2)
        plot_boundaries(matrix_y, norm_x, weights[3], 3)

        full_df = hstack((matrix_y, norm_x))
        separated_house = {id_houses[house]: full_df[full_df[:, 0] == house][:, 1:] for house in range(0, 4)}

        for house, h_df in zip(separated_house.keys(), separated_house.values()):
            scatter(
                h_df[:, 0],
                h_df[:, 1],
                color=houses_colors[house],
                alpha=0.7,
            )
        show()

    except FileNotFoundError:
        print_error("FileNotFoundError: provided file not found.")
    except PermissionError:
        print_error("PermissionError: permission denied on provided file.")
    except EmptyDataError:
        print_error("EmptyDataError: Provided dataset is empty.")
    except KeyError as err:
        print_error(f"KeyError: {err} is not in the required file.")

if __name__ == "__main__":
    main()