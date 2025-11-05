#!./.venv/bin/python

from pandas import DataFrame, read_csv
from numpy import ndarray, array, vstack, sum
from matplotlib.pyplot import subplots, show, Figure, xlabel, ylabel
from sys import argv
from dslr_lib.errors import print_error
from dslr_lib.maths import min, max, mean, std

error = {
    "ARG_ERR" : "Error: Invalid number of arguments.",
    "PARAM_ERR" : "Error: Invalid parameters.",
    "ALL_ERR": "Error: You can only use the \"All\" parameter alone."
}

houses_colors = {
    "Ravenclaw": "c",
    "Gryffindor": "r",
    "Slytherin": "g",
    "Hufflepuff": "y"
}

def setup_scatter(df: DataFrame) -> dict[str, DataFrame]:
    """
    Separate a DataFrame into sub-DataFrames, one for each Hogwarts house, containing only numerical columns.

    Args:
        df (DataFrame): Input DataFrame containing student data, including a "Hogwarts House" column.

    Returns:
        dict[str, DataFrame]: A dictionary where keys are Hogwarts house names and values are DataFrames
                              containing only numerical columns for students of that house.
    """
    hogwarts_house = df["Hogwarts House"]
    hogwarts_house_names = ["Ravenclaw", "Slytherin", "Hufflepuff", "Gryffindor"]
    return {name: df[hogwarts_house == name].select_dtypes(include="number") for name in hogwarts_house_names}

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

    number_df = df.select_dtypes(include="number")
    plot_dict: dict[str , tuple[Figure, ndarray]] = {
        name:
        subplots(
            4,
            3
        )
        for name in to_show
    }
    separated_house = setup_scatter(df)

    for x, plt in zip(plot_dict.keys(), plot_dict.values()):
        axs_index = 0
        df_drop_x = number_df.drop(x, axis=1)
        for i, y in enumerate(df_drop_x):
            plt[0].suptitle(x)
            actual_plt = plt[1][i % 4, axs_index]
            actual_plt.set_title(y)
            for house, h_df in zip(separated_house.keys(), separated_house.values()):
                actual_plt.scatter(
                    h_df[x],
                    h_df[y],
                    color=houses_colors[house],
                    alpha=0.7
                )
            if (i + 1) % 4 < i % 4:
                axs_index += 1
    show()
    return

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1 (ndarray): First point as a numpy array.
        point2 (ndarray): Second point as a numpy array.

    Returns:
        float: Euclidean distance between the two points.
    """
    return sum([(x + y) ** 2 for x, y in zip(point2, point1)]) ** 0.5

def find_similar(df: DataFrame):
    """
    Find the pair of numerical columns in the DataFrame with the smallest Euclidean distance after normalization.

    Args:
        df (DataFrame): Input DataFrame containing numerical columns.

    Returns:
        tuple[str, str]: Names of the two columns with the smallest Euclidean distance.
    """
    df_nona = df.select_dtypes(include="number").dropna()

    lowest_diff = (float("inf"), ("", ""))
    check_dict = {name: df_nona[name] for name in df_nona.keys()}

    for x_name, x_df in zip(check_dict.keys(), check_dict.values()):
        for y_name, y_df in zip(check_dict.keys(), check_dict.values()):
            if x_name == y_name:
                continue
            normalized_x = (x_df - mean(x_df)) / std(x_df)
            normalized_y = (y_df - mean(y_df)) / std(y_df)
            diff = euclidean_distance(normalized_x, normalized_y)
            if diff < lowest_diff[0]:
                lowest_diff = (diff, (x_name, y_name))
    return lowest_diff[1]

def show_similar(df: DataFrame) -> None:
    """
    Display a scatter plot of the two most similar numerical columns in the DataFrame,
    with points colored by Hogwarts house.

    Args:
        df (DataFrame): Input DataFrame containing student data, including a "Hogwarts House" column.

    Returns:
        None: Displays the scatter plot using matplotlib.
    """
    f, ax = subplots()

    separated_house = setup_scatter(df)

    l_diff_x, l_diff_y = find_similar(df)

    for house, h_df in zip(separated_house.keys(), separated_house.values()):
        ax.scatter(h_df[l_diff_x],
                   h_df[l_diff_y],
                   color=houses_colors[house],
                   alpha=0.5)
    f.suptitle("Similar Subjects")
    f.supxlabel(l_diff_x)
    f.supylabel(l_diff_y)
    show()

def main() -> None:
    """
    Main function to read a CSV file and display scatter plots of the data.

    Args:
        None: Expects a single command-line argument: the path to the CSV file.

    Returns:
        None: Displays scatter plots or prints an error message if arguments are invalid.
    """
    try:
        assert len(argv) >= 2, "ARG_ERR"
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1)

        if len(argv) == 2:
            show_similar(df)
            return
        subjects_name = df.select_dtypes(include="number").columns.tolist()
        subjects_name.append("All")
        for arg in argv[2:]:
            if arg == "All" and len(argv) != 3:
                raise AssertionError("ALL_ERR")
            assert arg in subjects_name, "PARAM_ERR"
        if "All" in argv[2:]:
            subjects_list = df.select_dtypes(include="number").columns.tolist()
        else:
            subjects_list = argv[2:]
        show_plots(df, subjects_list)
    except AssertionError as err:
        print_error(error[str(err)])


if __name__ == "__main__":
    main()