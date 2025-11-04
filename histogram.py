#!./.venv/bin/python
from numbers import Number

import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
from sys import argv
from numpy import ndarray
from typing import Any

from dslr_lib.errors import print_error
from dslr_lib.maths import total_count, count, mean, var, std, min, max, Q1, Q2, Q3, nan_count

error = {
    "ARG_ERR" : "Error: Invalid number of arguments"
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
    axis: Any
) -> tuple:
    subsets = generate_subsets(df, column)
    axis.hist(df[column].values, color="k", alpha=0.20)
    for i in range(4):
        axis.hist(subsets[i], color=houses_colors[id_houses[i]], alpha=0.8, label=id_houses[i])
    return subsets


def one_way_anova(
    infos: list[Any]
) -> tuple[Any]:
    def f_test(
        subset: list
    ) -> float:
        grand_mean = mean(subset[2])
        dfb = 4 - 1
        dfw = len(subset[2]) - 4
        ssb = 0
        for i in range(4):
            grp_mean = mean(subset[1][i])
            tmp = len(subset[1][i]) * (grp_mean - grand_mean) ** 2
            # tmp /= dfb
            ssb += tmp
        ssw = 0
        for i in range(4):
            tmp_sum = 0
            grp_mean = mean(subset[1][i])
            for x in subset[1][i]:
                tmp_sum += (x - grp_mean) ** 2 # / dfw
            ssw += tmp_sum
        return (ssb / dfb) / (ssw / dfw)
    min_f = -1
    min_f_idx = -1
    max_f = -1
    max_f_idx = -1
    for info in infos:
        f_result = f_test(info)
        print(f_result)
        if min_f > f_result or min_f == -1:
            min_f = f_result
            min_f_idx = info[0]
        if f_result > max_f:
            max_f = f_result
            max_f_idx = info[0]
    print(min_f, min_f_idx)
    print(max_f, max_f_idx)



def find_uniform_distribution(
    df: DataFrame
):
    # df.sort_values(by="Hogwarts House", axis=0)
    _, ax = plt.subplots(3, 5)
    all_subsets = list()
    for y in range(3):
        for x in range(5):
            if 5 + y * 5 + x >= len(df.columns):
                break
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
    uniform_scores = one_way_anova(all_subsets)
    plt.show()


def main():
    try:
        assert len(argv) == 2, "ARG_ERR"
        df: DataFrame = read_csv(argv[1], header=0).drop("Index", axis=1)
        find_uniform_distribution(df)
    except AssertionError as err:
        print_error(error[str(err)])

if __name__ == "__main__":
    main()