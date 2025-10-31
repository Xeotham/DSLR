from sys import stderr

def print_error(*args, **kwargs):
    """Prints a message or multiple messages to stderr (standard error).

    This function is a wrapper around the built-in `print` function,
    redirecting output to stderr instead of stdout. It accepts the same
    arguments as `print`.

    Args:
        *args: Variable length argument list. The objects to print.
            Each object is converted to a string and written to stderr.
        **kwargs: Arbitrary keyword arguments. Commonly used kwargs include:
            - sep (str): String inserted between values. Defaults to ' '.
            - end (str): String appended after the last value. Defaults to '\n'.
            - flush (bool): Whether to forcibly flush the stream. Defaults to False.

    Returns:
        None: This function does not return any value.

    Example:
        >>> print_error("An error occurred!", "Please check your input.")
        An error occurred! Please check your input.
    """

    print(*args, file=stderr, **kwargs)
