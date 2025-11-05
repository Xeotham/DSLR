#!./.venv/bin/python
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
from sys import argv
from typing import Any

from dslr_lib.errors import print_error
from dslr_lib.maths import mean

error = {
    "ARG_ERR": "Error: Invalid number of arguments"
}

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


def generate_subsets(
    df: DataFrame,
    column: str
) -> tuple:
    return (
        df[df["Hogwarts House"] == "Ravenclaw"][column].dropna(),
        df[df["Hogwarts House"] == "Gryffindor"][column].dropna(),
        df[df["Hogwarts House"] == "Slytherin"][column].dropna(),
        df[df["Hogwarts House"] == "Hufflepuff"][column].dropna(),
    )


def generate_histogram(
    df: DataFrame,
    column: str,
    graph: Any
) -> tuple:
    """
    Generates one histogram in the passed graph.
    The data is fetched in the dataframe 'df' at column 'column'.

    Args:
        df: The dataset
        column: The column we want to plot
        graph: The graph object that we can use to show the histogram

    Returns:
        tuple: Returns the subset of the data split by houses
    """
    subsets = generate_subsets(df, column)
    graph.hist(df[column].values, color="k", alpha=0.20)
    for i in range(4):
        graph.hist(
            subsets[i],
            color=houses_colors[id_houses[i]],
            alpha=0.8,
            label=id_houses[i]
        )
    graph.set_title(column)
    return subsets


def one_way_anova(
    infos: list
) -> tuple:
    """
    Uses inner function f_test to check the ratio between the variance
    between groups over the variance within the groups.
    The function tries to find the course whose houses results are the
    most similar.

    Args:
        infos: An array of tuple each containing the index's column data,
            four subsets of the data filtered by house,
            the whole data unfiltered

    Returns:
        tuple: Return the tuple whose f-value is the least amongst them all
    """
    def f_test(
        info_tuple: list
    ) -> float:
        """
        Calculates the F-value of a subset: the ratio of the variance between
        the groups over the variance within the group

        Args:
            info_tuple: The tuple containing the data of the course
                index of column, data split by house, unsplit data

        Returns:
            float: F-value as a float
        """
        grand_mean = mean(info_tuple[2])
        dfb = 4 - 1
        dfw = len(info_tuple[2]) - 4
        ssb = 0
        for i in range(4):
            grp_mean = mean(info_tuple[1][i])
            tmp = len(info_tuple[1][i]) * (grp_mean - grand_mean) ** 2
            ssb += tmp
        ssw = 0
        for i in range(4):
            tmp_sum = 0
            grp_mean = mean(info_tuple[1][i])
            for x in info_tuple[1][i]:
                tmp_sum += (x - grp_mean) ** 2
            ssw += tmp_sum
        return (ssb / dfb) / (ssw / dfw)

    min_f = -1
    min_ret = ()
    for info in infos:
        f_result = f_test(info)
        if min_f > f_result or min_f == -1:
            min_f = f_result
            min_ret = info
    return min_ret[0], min_ret[1], min_ret[2]


def show_histograms(
    df: DataFrame
) -> None:
    """
    Shows the histograms of all the courses

    Args:
        df: the data containing all the courses

    Returns:
        None: Nothing
    """
    fg, ax = plt.subplots(3, 5)
    fg.suptitle("Histogram")
    all_subsets = list()
    for y in range(3):
        for x in range(5):
            if 5 + y * 5 + x >= len(df.columns):
                fg.delaxes(ax[y, x])
                continue
            new_subset = generate_histogram(
                df,
                df.columns[5 + y * 5 + x],
                ax[y, x],
            )
            all_subsets.append([
                y * 5 + x,
                new_subset,
                df[df.columns[5 + y * 5 + x]].dropna(),
            ])
    idx, _, __ = one_way_anova(all_subsets)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[idx // 5, idx % 5].spines[axis].set_linewidth(2.5)
        ax[idx // 5, idx % 5].spines[axis].set_color('m')
    plt.show()


def main() -> None:
    """
    The main function

    Returns:
        None: Nothing
    """
    try:
        assert len(argv) == 2, "ARG_ERR"
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1)
        show_histograms(df)
    except AssertionError as err:
        print_error(error[str(err)])


if __name__ == "__main__":
    main()
