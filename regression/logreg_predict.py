#!../.venv/bin/python
import sys
sys.path.insert(1, "..")
sys.path.insert(2, ".")
# sys.path.insert(3, "./visualization")

import numpy as np
from sys import argv
from numpy import ndarray, vectorize, array, zeros, argmax
from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError

from dslr_lib.errors import print_error
from dslr_lib.maths import normalize
from dslr_lib.regressions import predict_proba
from regression.logreg_train import prepare_dataset


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

def load_parameters(
) -> tuple[ndarray[float], ndarray[float], ndarray[float], ndarray[float]]:
    df = read_csv("../datasets/weights.csv")
    return (
        array([df["ravenclaw"].values]).T,
        array([df["slytherin"].values]).T,
        array([df["gryffindor"].values]).T,
        array([df["hufflepuff"].values]).T,
    )


def prepare_predictions(
    test_path: str
) -> ndarray:
    test_df: DataFrame = read_csv(test_path, header=0).drop("Index", axis=1)
    _, test_x = prepare_dataset(test_df, fill=True)
    train_df: DataFrame = read_csv("../datasets/dataset_train.csv", header=0).drop("Index", axis=1)
    _, train_x = prepare_dataset(train_df)
    return normalize(test_x, train_x)


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


def main():
    try:
        assert len(argv) == 2
        test_x = prepare_predictions(argv[1])
        matrix_t = load_parameters()
        houses = generate_predictions(test_x, matrix_t)
        house_map = np.vectorize(lambda x: id_houses[x])
        houses_csv = DataFrame(
            data=house_map(houses),
            columns=["Hogwarts House"],
        )
        project_path = str(__file__)
        project_path = project_path[:-len("/logreg_predict.py")]
        path = "../datasets/houses.csv"
        if not path.startswith('/'):
            path = "/" + path
        path = project_path + path
        houses_csv.to_csv(path, index_label="Index")

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