import numpy as np
from minimask.mask import Mask
from minimask.spherical_poly import spherical_polygon


def check(a):
    assert a


def test_mask_sample():
    """ """
    vertices = [[0,0],[10,0],[10,10],[0,10]]
    S = spherical_polygon(vertices)

    M = Mask(polys=[S, S])

    x,y = M.sample(n=1000)

    yield check, len(x) == 1000
    yield check, len(y) == 1000

    yield check, np.abs(x.min()) < 1
    yield check, np.abs(y.min()) < 1
    yield check, np.abs(x.max() - 10) < 1
    yield check, np.abs(y.max() - 10) < 1

    r = M.contains(x, y)
    yield check, np.sum(r) == len(x)

    inside, w = M.get_weight(x, y)
    yield check, np.allclose(w, np.ones((len(w), 2)))

    inside, w = M.get_combined_weight(x, y, operation='sum')
    yield check, np.allclose(w, 2 * np.ones(len(w)))

    x, y = M.sample(density=1000)
    yield check, np.abs(len(x) - 100000) < 1000


def test_mask():
    """ check empty mask """
    M = Mask()
    x, y = M.sample(n=1000)

    yield check, len(x) == 1000
    yield check, len(y) == 1000

    yield check, np.abs(x.min()) < 10
    yield check, np.abs(y.min() + 90) < 10
    yield check, np.abs(x.max() - 360) < 10
    yield check, np.abs(y.max() - 90) < 10

def test_undefined():
    M = Mask()
    x, y = M.sample(n=0)
    yield check, len(x) == 0

    x, y = M.sample(density=0)
    yield check, len(x) == 0

    try:
        x, y = M.sample()
    except ValueError:
        pass

    try:
        x, y = M.sample(n='hi')
    except ValueError:
        pass

    try:
        x, y = M.sample(density='hi')
    except ValueError:
        pass

    try:
        x, y = M.sample(n=-1)
    except ValueError:
        pass

    try:
        x, y = M.sample(density=-1)
    except ValueError:
        pass

    try:
        x, y = M.sample(density=float('nan'))
    except ValueError:
        pass

    try:
        x, y = M.sample(n=float('nan'))
    except ValueError:
        pass


