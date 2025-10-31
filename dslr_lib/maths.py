from pandas import DataFrame
from numpy import sum, var

def total_count(df: DataFrame) -> int:
    """
    Calculate the total number of rows in a DataFrame, including NaN values.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        int: Total number of rows in the DataFrame.
    """
    return df.shape[0]

def count(df: DataFrame) -> int:
    """
    Calculate the number of non-NaN rows in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        int: Number of non-NaN rows in the DataFrame.
    """
    df_nona = df[df.notna()]
    return df_nona.shape[0]

def mean(df: DataFrame) -> float:
    """
    Calculate the arithmetic mean of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Arithmetic mean of non-NaN values.
    """
    df_nona = df[df.notna()]
    return sum(df_nona.values) / count(df_nona)

def var(df: DataFrame) -> float:
    """
    Calculate the variance of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Variance of non-NaN values.
    """
    df_nona = df[df.notna()]
    df_mean = mean(df)
    return sum([(i - df_mean) ** 2 for i in df_nona.values]) / count(df_nona)

def std(df: DataFrame) -> float:
    """
    Calculate the standard deviation of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Standard deviation of non-NaN values.
    """
    return var(df) ** 0.5

def min(df: DataFrame) -> float:
    """
    Find the minimum value among non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Minimum value in the DataFrame.
    """
    df_nona = df[df.notna()]
    df_min = float("inf")
    for i in df_nona.values:
        if i < df_min:
            df_min = i
    return df_min

def max(df: DataFrame) -> float:
    """
    Find the maximum value among non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Maximum value in the DataFrame.
    """
    df_nona = df[df.notna()]
    df_max = float("-inf")
    for i in df_nona.values:
        if i > df_max:
            df_max = i
    return df_max

def test_quartile(Q: float, arg_list: list) -> float:
    """
    Calculate the value of a quartile based on its position and a list of values.

    Args:
        Q (float): Quartile position (e.g., 1.25 for Q1).
        arg_list (list): List of sorted values.

    Returns:
        float: Value of the quartile.
    """
    Rinf = arg_list[int(Q.__floor__())]
    Rsup = arg_list[int(Q.__ceil__())]
    final_value = 0
    if not Q.is_integer():
        if Q % 1 == 0.25:
            final_value = ((Rinf * 3) + Rsup) / 4
        elif Q % 1 == 0.5:
            final_value = (Rinf + Rsup) / 2
        elif Q % 1 == 0.75:
            final_value = (Rinf + (Rsup * 3)) / 4
        else:
            final_value = arg_list[int(Q.__floor__())]
    return final_value

def Q1(df: DataFrame) -> float:
    """
    Calculate the first quartile (Q1) of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: First quartile (Q1) value.
    """
    df_nona = df[df.notna()].sort_values()
    df_q1 = ((len(df_nona) + 3) / 4)
    return test_quartile(df_q1 - 1, df_nona.to_list())

def Q2(df: DataFrame) -> float:
    """
    Calculate the second quartile (Q2, median) of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Second quartile (Q2) value.
    """
    df_nona = df[df.notna()].sort_values()
    df_q2 = (((len(df_nona)) + 1) / 2)
    return test_quartile(df_q2 - 1, df_nona.to_list())

def Q3(df: DataFrame) -> float:
    """
    Calculate the third quartile (Q3) of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Third quartile (Q3) value.
    """
    df_nona = df[df.notna()].sort_values()
    df_q3 = (((3 * len(df_nona)) + 1) / 4)
    return test_quartile(df_q3 - 1, df_nona.to_list())

def nan_count(df: DataFrame) -> int:
    """
    Count the number of NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        int: Number of NaN values in the DataFrame.
    """
    return sum(1 for x in df.values.flatten() if x != x)
