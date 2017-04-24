import numpy as np
from minimask.mask import Mask
from minimask.spherical_poly import spherical_polygon


def check(a):
    assert a


def test_mask_sample():
    """ """
    vertices = [[0,0],[10,0],[10,10],[0,10]]
    S = spherical_polygon(vertices)

    M = Mask(polys=[S], fullsky=False)

    x,y = M.sample(100)

    yield check, len(x) == 1000
    yield check, len(y) == 1000

    yield check, np.abs(x.min()) < 1
    yield check, np.abs(y.min()) < 1
    yield check, np.abs(x.max() - 10) < 1
    yield check, np.abs(y.max() - 10) < 1

    r = M.contains(x, y)
    yield check, np.sum(r) == len(x)

    w = M.get_weight(x, y)
    yield check, np.allclose(w, np.ones(len(w)))
