#!../.venv/bin/python3
from sys import argv, path
path.append("..")
path.append(".")
path.append("./analysis")

from pandas import DataFrame, read_csv
from pandas.errors import EmptyDataError
from numpy import array, vstack
from dslr_lib.errors import print_error
from dslr_lib.maths import total_count, count, mean, var, std, min, max, Q1, Q2, Q3, nan_count

def describe(df: DataFrame):
    """
    Computes and returns descriptive statistics for each column in the input DataFrame.
    The statistics include total count, count (non-NaN), mean, variance, standard deviation,
    minimum, maximum, quartiles (25%, 50%, 75%), and NaN count.

    Args:
        df (DataFrame): Input DataFrame for which to compute descriptive statistics.

    Returns:
        Tuple[DataFrame, DataFrame]:
            - A custom DataFrame with detailed descriptive statistics.
            - The standard pandas `describe()` output for comparison.

    Notes:
        - Uses custom functions `count`, `mean`, `var`, `std`, `min`, `max`, `Q1`, `Q2`, `Q3`, and `nan_count`.
        - The custom DataFrame includes additional statistics like total count and NaN count.
    """
    # List of statistical operations to apply
    oper_lst = [count, mean, var, std, min, max, Q1, Q2, Q3]
    # Names for the rows in the output DataFrame
    oper_names = [
        "Total count", "Count", "Mean", "Variance", "Std Dev",
        "Min", "Max", "25%", "50%", "75%", "NaN Count"
    ]

    # Initialize the result array with the total count for each column
    describe_arr = array([[total_count(df[i]) for i in df.keys()]])

    # Compute and stack each statistic for all columns
    for f in oper_lst:
        describe_arr = vstack((
            describe_arr,
            array([[f(df[i].dropna().values) for i in df.keys()]])
        ))

    # Add NaN count for each column
    describe_arr = vstack((
        describe_arr,
        array([[nan_count(df[i].values) for i in df.keys()]])
    ))

    # Convert the result array to a DataFrame and rename rows and columns
    describe_df = DataFrame(describe_arr).rename(
        index={i: name for i, name in enumerate(oper_names)},
        columns={i: name for i, name in enumerate(df.keys())}
    )

    # Print the custom descriptive statistics DataFrame
    print(describe_df)

    # Return both the custom and standard pandas describe DataFrames
    return describe_df, df.describe()


def main():
    try:
        assert len(argv) == 2
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1).select_dtypes(include="number")
        describe(df)
    except AssertionError:
        print_error("Please pass a dataset as parameter")
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