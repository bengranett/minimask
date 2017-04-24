import numpy as np
import logging


def invsquare(x):
    """ Sum inverse of squares. """
    x = np.array(x)
    nonzero = x != 0
    if np.sum(nonzero) == 0:
        return 0
    return np.sum(1. / x[nonzero]**2)


operations = {
    'and': np.logical_and.reduce,
    'sum': np.sum,
    'square': lambda x: np.sum(np.square(x)),
    'mean': np.mean,
    'invsquare': invsquare
}


def combine(weights, operation='sum'):
    """ Combine a set of weights.

    Parameters
    ----------
    weights : list
        a list of weight values
    operation : str or callable
        the operation to apply

    Returns
    -------
    float : weight value
    """
    single = False

    try:
        v = weights[0][0]
    except TypeError:
        weights = [weights]
        single = True

    if hasattr(operation, '__call__'):
        op = operation
    else:
        op = operations[operation.lower()]

    out = [op(w) for w in weights]

    if single:
        return out[0]
    else:
        return out
