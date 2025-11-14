#!../.venv/bin/python
import sys
from sys import argv


sys.path.insert(1, "..")
sys.path.insert(2, ".")
sys.path.insert(3, "../visualization")

from numpy import ndarray, vectorize, array, zeros, argmax
from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from pandas.api.types import is_numeric_dtype
from dslr_lib.maths import normalize, mean
from dslr_lib.regressions import gradient_descent, predict, predict_proba, sigmoid
from dslr_lib.errors import print_error
from dslr_lib.opti_bonus import cross_validation
from matplotlib.pyplot import figure, plot, scatter, show, legend, xlim, ylim, fill_between

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

# TODO: Place on dslr_lib
def prepare_dataset(
    df: DataFrame,
    fill: bool = False
) -> tuple[ndarray[int], ndarray[float]]:
    if fill:
        for i in range(0, len(df.columns)):
            if not is_numeric_dtype(df[df.columns[i]]):
                continue
            df[df.columns[i]] = df[df.columns[i]].fillna(df[df.columns[i]].mean())
    else:
        df.dropna(inplace=True)
    matrix_y = df[df.columns[0]]
    if "Hogwarts House" in df.columns:
        matrix_y = df["Hogwarts House"].map(houses_id)
    matrix_x = df.select_dtypes(include="number")
    matrix_x = matrix_x[[
        "Herbology",
        "Defense Against the Dark Arts"
    ]]
    return matrix_y.values, matrix_x.values


def regression_wrapper(
    matrix_y: ndarray[int],
    matrix_x: ndarray[float],
    houses: int
) -> ndarray[float]:
    train_y = matrix_y.copy()
    train_y[matrix_y != houses] = 0
    train_y[matrix_y == houses] = 1
    weights = gradient_descent(matrix_x, train_y, max_iter=1000, alpha=0.01)

    return weights.flatten()

# TODO: Place on dslr_lib or on bonus.
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
    # plot(xd, yd, 'k', ls='--', color=houses_colors[id_houses[houses]])
    # fill_between(xd, yd, ymin, color=houses_colors[id_houses[houses]], label=id_houses[houses], alpha=0.2)


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

    # TODO: Add parameter to eventually show it
    # plot_boundaries(matrix_y, norm_x, t_ravenclaw, 0)
    # plot_boundaries(matrix_y, norm_x, t_slytherin, 1)
    # plot_boundaries(matrix_y, norm_x, t_gryffindor, 2)
    # plot_boundaries(matrix_y, norm_x, t_hufflepuff, 3)

    # full_df = hstack((matrix_y, norm_x))
    # separated_house = {id_houses[house]: full_df[full_df[:, 0] == house][:, 1:] for house in range(0, 4)}
    #
    # # scatter(matrix_x[:, 0], matrix_x[:, 1])
    # for house, h_df in zip(separated_house.keys(), separated_house.values()):
    #     scatter(
    #         h_df[:, 0],
    #         h_df[:, 1],
    #         color=houses_colors[house],
    #         alpha=0.7,
    #     )
    # legend()
    # show()

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


from sklearn.metrics import accuracy_score

def main():
    try:
        assert len(argv) == 2
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1)
        matrix_y, matrix_x = prepare_dataset(df)
        matrix_y.resize((matrix_y.shape[0], 1))
        thetas_weights = logreg_train(matrix_y, matrix_x)

        p_score = cross_validation(matrix_x, matrix_y)
        print(f"Accuracy: {p_score}")

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



    except AssertionError:
        print_error("Please pass a \"dataset_train.csv\" as parameter")
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