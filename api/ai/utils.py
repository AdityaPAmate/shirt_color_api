####################################################################
# EXECUTION TIME LOGGER (For debugging - logs function execution time)
####################################################################

import time
import functools
import logging

logger = logging.getLogger(__name__)


def log_execution_time(func):
    """
    Decorator: Measures how many seconds a function takes to execute
    and logs the execution time.

    Usage
    -----
        @log_execution_time
        def some_function(...):
            ...

    This does not change the function's internal logic.
    It only measures the execution time from outside.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        logger.info(
            f"[TIMER] {func.__qualname__} took {elapsed:.4f} sec"
        )

        return result

    return wrapper