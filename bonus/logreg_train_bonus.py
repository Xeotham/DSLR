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
from dslr_lib.errors import print_error
from dslr_lib.opti_bonus import cross_validation, logreg_train, FeaturesSelector
from dslr_lib.regressions import houses_id, id_houses, houses_colors


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
    return matrix_y.values, matrix_x.values

def features_name(features: ndarray, df: DataFrame) -> list:
    names = []

    def cmp_features(cmp_1, cmp_2):
        for v_1, v_2 in zip(cmp_1, cmp_2):
            if v_1 != v_2:
                return False
        return True

    for feature in features.T:
        for name, values in zip(df.keys(), df.values.T):
            if cmp_features(feature, values):
                names.append(name)
                break
    return names

def main():
    try:
        assert len(argv) == 2
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1)
        matrix_y, matrix_x = prepare_dataset(df)
        matrix_y.resize((matrix_y.shape[0], 1))
        thetas_weights = logreg_train(matrix_y, matrix_x)

        features_select = FeaturesSelector(matrix_x, matrix_y, 4, 0.98)
        matrix_x = features_select.find_best_features()
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