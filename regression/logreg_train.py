#!../.venv/bin/python
import sys
from sys import argv

from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError

sys.path.insert(0, "..")
from dslr_lib.errors import print_error



def main():
    try:
        assert len(argv) == 2
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1)

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