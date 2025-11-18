from multiprocessing import Process


def threaded(func):
    """
    Decorator that multithreads the target function
    with the given parameters. Returns the thread
    created for the function
    """
    def wrapper(*args, **kwargs):
        thread = Process(target=func, args=args, daemon=True)
        thread.start()
        return thread
    return wrapper

