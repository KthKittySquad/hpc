import timeit
from functools import wraps
from typing import Callable
import numpy as np


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        arr = np.zeros((10,))
        result = None
        for n in range(10):
            start = timeit.default_timer()
            result = func(*args, **kwargs)
            end = timeit.default_timer()
            arr[n] = end - start
        print(f"Function: {func.__name__}")
        print(f"Average Execution Time: {arr.mean():.6f} seconds")
        print(f"Standard Deviation: {arr.std():.6f} seconds")
        return result

    return wrapper
