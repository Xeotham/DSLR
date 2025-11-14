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
from dslr_lib.opti_bonus import cross_validation, logreg_train
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