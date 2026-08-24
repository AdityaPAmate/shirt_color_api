####################################################################
# EXECUTION TIME LOGGER (डिबगिंगसाठी — कुठला function किती वेळ
# घेतो हे टर्मिनलवर लॉग करण्यासाठी)
####################################################################

import time
import functools


def log_execution_time(func):
    """
    Decorator: कुठल्याही function ला हा लावला की, तो function
    चालायला किती सेकंद लागले हे टर्मिनलवर प्रिंट होतं.

    Usage
    -----
        @log_execution_time
        def some_function(...):
            ...

    हे function ची body अजिबात बदलत नाही -- फक्त बाहेरून वेळ मोजतं.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        print(f"[TIMER] {func.__qualname__} took {elapsed:.4f} sec")

        return result

    return wrapper