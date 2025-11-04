#!./.venv/bin/python

from pandas import DataFrame, read_csv
from numpy import ndarray, array, vstack
from sys import argv
from dslr_lib.errors import print_error
from dslr_lib.maths import total_count, count, mean, var, std, min, max, Q1, Q2, Q3, nan_count

error = {
    "ARG_ERR" : "Error: Invalid number of arguments"
}

def describe(df: DataFrame):
    oper_lst = [count, mean, var, std, min, max, Q1, Q2, Q3, nan_count]
    oper_names = ["Total count", "Count", "Mean", "Variance", "Std Dev", "Min", "Max", "25%", "50%", "75%", "NaN Count"]
    describe_arr = array([[total_count(df[i]) for i in df.keys()]])
    for f in oper_lst:
        describe_arr = vstack((describe_arr, array([[f(df[i]) for i in df.keys()]])))
    describe_df = DataFrame(describe_arr).rename(index={i: name for i, name in enumerate(oper_names)},
                                                 columns={i:name for i, name in enumerate(df.keys())})
    print(describe_df)
    print(df.describe())


def main():
    try:
        assert len(argv) == 2, "ARG_ERR"
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1).select_dtypes(include="number")
        describe(df)
    except AssertionError as err:
        print_error(error[str(err)])

if __name__ == "__main__":
    main()