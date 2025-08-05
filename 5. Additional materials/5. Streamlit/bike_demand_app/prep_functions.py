import pandas as pd
import numpy as np
from functools import wraps
import time
import re

def prep_logger(func):
    """
    a wrapper function that records preprocessed objects' shape changes and the time that preprocessing takes
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        shape_before = args[0].shape
        shape_after = result.shape
        print(f"{func.__name__} => took: {round(end_time - start_time, 2)} seconds; shape before:{shape_before} shape after:{shape_after}")
        return result
    return wrapper
    
    return wrapper

