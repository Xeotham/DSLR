#!../.venv/bin/python
import sys
from sys import argv
sys.path.insert(1, "..")
sys.path.insert(2, ".")
# sys.path.insert(3, "./visualization")

from numpy import ndarray, vectorize, array
from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError

from dslr_lib.errors import print_error
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
    return df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3]


def main():
    try:
        assert len(argv) == 2
        test_df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1)
        test_x = test_df.select_dtypes(include="number")
        test_x = test_x[[
            "Herbology",
            "Astronomy",
            "Ancient Runes",
            "Defense Against the Dark Arts"
        ]]
        ravenclaw_t, slytherin_t, gryffindor_t, hufflepuff_t = load_parameters()
        train_df: DataFrame = read_csv("../datasets/dataset_train.csv", header=0).drop("Index", axis=1)
        _, train_x = prepare_dataset(train_df)






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