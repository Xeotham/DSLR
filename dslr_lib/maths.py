from pandas import DataFrame
from numpy import ndarray, sum

def total_count(df: ndarray) -> int:
    """
    Calculate the total number of rows in a DataFrame, including NaN values.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        int: Total number of rows in the DataFrame.
    """
    return df.shape[0]

def count(df: ndarray) -> int:
    """
    Calculate the number of non-NaN rows in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        int: Number of non-NaN rows in the DataFrame.
    """
    # df_nona = df[df.notna()]
    return df.shape[0]

def mean(df: ndarray) -> float:
    """
    Calculate the arithmetic mean of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Arithmetic mean of non-NaN values.
    """
    # df_nona = df[df.notna()]
    if count(df) == 0:
        return 0
    return sum(df) / count(df)

def var(df: ndarray) -> float:
    """
    Calculate the variance of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Variance of non-NaN values.
    """
    # df_nona = df[df.notna()]
    if count(df) == 0:
        return 0
    df_mean = mean(df)
    return sum([(i - df_mean) ** 2 for i in df]) / count(df)

def std(df: ndarray) -> float:
    """
    Calculate the standard deviation of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Standard deviation of non-NaN values.
    """
    return var(df) ** 0.5

def min(df: ndarray) -> float:
    """
    Find the minimum value among non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Minimum value in the DataFrame.
    """

    if len(df) < 1:
        return float("nan")

    df_min = float("inf")
    for i in df:
        if i < df_min:
            df_min = i
    return df_min

def max(df: ndarray) -> float:
    """
    Find the maximum value among non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Maximum value in the DataFrame.
    """
    if len(df) < 1:
        return float("nan")

    df_max = float("-inf")
    for i in df:
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

    if len(arg_list) < 1:
        return float("nan")

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
    else:
        final_value = arg_list[int(Q)]
    return final_value

def Q1(df: ndarray) -> float:
    """
    Calculate the first quartile (Q1) of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: First quartile (Q1) value.
    """
    sorted_df = df.copy()
    sorted_df.sort()
    df_q1 = ((len(sorted_df) + 3) / 4)
    return test_quartile(df_q1 - 1, sorted_df.tolist())

def Q2(df: ndarray) -> float:
    """
    Calculate the second quartile (Q2, median) of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Second quartile (Q2) value.
    """
    sorted_df = df.copy()
    sorted_df.sort()
    df_q2 = (((len(sorted_df)) + 1) / 2)
    return test_quartile(df_q2 - 1, sorted_df.tolist())

def Q3(df: ndarray) -> float:
    """
    Calculate the third quartile (Q3) of non-NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        float: Third quartile (Q3) value.
    """
    sorted_df = df.copy()
    sorted_df.sort()
    df_q3 = (((3 * len(sorted_df)) + 1) / 4)
    return test_quartile(df_q3 - 1, sorted_df.tolist())

def nan_count(df: ndarray) -> int:
    """
    Count the number of NaN values in a DataFrame.

    Args:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        int: Number of NaN values in the DataFrame.
    """
    return sum(1 for x in df.flatten() if x != x)

def unnormalize(value, base):
    """
    Function to unnormalize a value.
    :param value: Value to unnormalize
    :param base: List to unnormalize with.
    :return: Unnormalized value.
    """
    return (value * std(base)) + mean(base)


def normalize(value, base):
    """
    Function to normalize a value based on a list.
    :param value: Value to normalize.
    :param base: List to normalize with.
    :return: Normalized value.
    """
    return (value - mean(base)) / std(base)
