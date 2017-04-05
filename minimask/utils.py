import numpy as np

def linspacing(low, high, step, nmin=3):
    """ """
    ext = high - low
    n = max(nmin, int(ext * 1. / step))
    return np.linspace(low, high, n)
