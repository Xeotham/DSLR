#!../.venv/bin/python
import sys
from sys import argv
sys.path.insert(1, "..")
sys.path.insert(2, ".")
# sys.path.insert(3, "./visualization")


from numpy import ndarray, vectorize, array
from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from pandas.api.types import is_numeric_dtype

from dslr_lib.maths import normalize, mean
from dslr_lib.regressions import gradient_descent, predict, predict_proba
from dslr_lib.errors import print_error
from matplotlib.pyplot import figure, plot, scatter, show, legend, xlim, ylim, fill_between

houses_colors = {
    "Ravenclaw": "c",
    "Gryffindor": "r",
    "Slytherin": "g",
    "Hufflepuff": "y"
}

houses_id = {
    "Ravenclaw": 0,
    "Gryffindor": 1,
    "Slytherin": 2,
    "Hufflepuff": 3
}

id_houses = {
    0: "Ravenclaw",
    1: "Gryffindor",
    2: "Slytherin",
    3: "Hufflepuff",
}

def prepare_dataset(
    df: DataFrame
) -> tuple[ndarray[int], ndarray[float]]:
    for i in range(0, len(df.columns)):
        if not is_numeric_dtype(df[df.columns[i]]):
            continue
        df[df.columns[i]] = df[df.columns[i]].fillna(df[df.columns[i]].mean())
    matrix_y = df[df.columns[0]]
    if "Hogwarts House" in df.columns:
        matrix_y = df["Hogwarts House"].map(houses_id)
    matrix_x = df.select_dtypes(include="number")
    matrix_x = matrix_x[[
        "Herbology",
        "Astronomy",
        "Ancient Runes",
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
    weights = gradient_descent(matrix_x, train_y)
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


def main():
    try:
        assert len(argv) == 2
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1)
        matrix_y, matrix_x = prepare_dataset(df)
        matrix_y.resize((matrix_y.shape[0], 1))
        thetas_weights = logreg_train(matrix_y, matrix_x)
        thetas_csv = DataFrame(
            data=array(thetas_weights).T,
            dtype=float,
            columns=["ravenclaw", "slytherin", "gryffindor", "hufflepuff"]
        )
        project_path = str(__file__)
        project_path = project_path[:-len("/logreg_train.py")]
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