import numpy as np

def linspacing(low, high, step, nmin=1):
    """ """
    ext = high - low
    if step == 0:
        n = nmin
    else:
        n = max(nmin, int(ext * 1. / step))
    return np.linspace(low, high, n+1)
