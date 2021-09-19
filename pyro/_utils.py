import numpy as np

"""
Internal utility functions
"""

def to_2D_arr(arr):
    arr = np.asanyarray(arr)
    if arr.ndim == 2:
        return arr

    if arr.ndim == 1:
        return arr[np.newaxis]
    elif arr.ndim == 0:
        return arr[np.newaxis, np.newaxis]
    else:
        raise ValueError(
            "Cannot expand array with %d dimensions to 2-D" % (arr.ndim)
        )