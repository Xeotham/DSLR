#!../.venv/bin/python
import sys
sys.path.insert(1, "..")
sys.path.insert(2, ".")
sys.path.insert(3, "./visualization")

from pandas import DataFrame, read_csv
from pandas.errors import EmptyDataError
from matplotlib.pyplot import subplots, show
from sys import argv
from histogram import generate_histogram
from scatter_plot import generate_scatter
from dslr_lib.errors import print_error

error = {
    "ARG_ERR" : "Error: Invalid number of arguments.",
    "PARAM_ERR" : "Error: Invalid parameters.",
    "ALL_ERR": "Error: You can only use the \"All\" parameter alone."
}

houses_colors = {
    "Ravenclaw": "c",
    "Gryffindor": "r",
    "Slytherin": "g",
    "Hufflepuff": "y",
    "All": "b"
}

def show_plots(df: DataFrame, to_show: list) -> None:
    """
    Generate and display a grid of scatter plots for all pairs of numerical columns in the DataFrame,
    with points colored by Hogwarts house.

    Args:
        to_show:
        df (DataFrame): Input DataFrame containing student data, including a "Hogwarts House" column.

    Returns:
        None: Displays the scatter plot grid using matplotlib.
    """

    fg, axs = subplots(len(to_show), len(to_show))

    for i, x in enumerate(to_show):
        for j, y in enumerate(to_show):
            actual_plt = axs[i, j]
            if i == j:
                generate_histogram(df, x, axs[i, j])
                axs[i, j].set_title("")
            else:
                generate_scatter(df, x, y, actual_plt, s_size=0.5)
            if j == 0:
                actual_plt.set_ylabel(to_show[i][: 4] + ".", fontsize=10)
            else:
                actual_plt.set_yticks([])
            if i == len(to_show) - 1:
                actual_plt.set_xlabel(to_show[j][: 4] + ".", fontsize=10)
            else:
                actual_plt.set_xticks([])
    fg.legend(["Ravenclaw",
            "Gryffindor",
            "Slytherin",
            "Hufflepuff"])
    fg.suptitle("Subjects Pair Plot")
    show()
    return

def main() -> None:
    """
    Main function to read a CSV file and display scatter plots of the data.

    Args:
        None: Expects a single command-line argument: the path to the CSV file.

    Returns:
        None: Displays scatter plots or prints an error message if arguments are invalid.
    """
    try:
        df: DataFrame = read_csv("../datasets/dataset_train.csv", header=0).drop("Index", axis=1)

        subjects_name = df.select_dtypes(include="number").columns.tolist()
        for arg in argv[1:]:
            assert arg in subjects_name, "PARAM_ERR"
        if len(argv) == 1:
            subjects_list = df.select_dtypes(include="number").columns.tolist()
        else:
            subjects_list = argv[1:]
        show_plots(df, subjects_list)
    except AssertionError as msg:
        print(str(msg))
        return
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