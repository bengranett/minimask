
import numpy as np
import minimask.weight_watcher as weight_watcher


def check(a, b):
    assert np.allclose(a, b)


def test_combine():
    cases = [
        ('sum', [1, 1, 1, 1], 4),
        ('SUM', [-1, 1, 1, -1], 0),
        ('square', [-1, 1, 1, -1], 4),
        ('invsquare', [-1, 1, 1, -1], 4),
        ('invsquare', [-1, 1, 1, 0], 3),
        ('mean', [1, 1, 1, 1], 1),
        ('and', [1, 1, 1, 1], 1),
        ('AND', [1, 1, 1, 1], 1),
        ('AND', [1, 1, 0, 1], 0),
        ('and', [[1,1],[0,1],[-1,0]], [1,0,0]),
        (np.sum, [1, 1, 1, 1], 4),

    ]

    for op, weights, r in cases:
        o = weight_watcher.combine(weights, op)
        yield check, r, o
