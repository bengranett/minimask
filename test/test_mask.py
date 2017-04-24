import numpy as np
from minimask.mask import Mask
from minimask.spherical_poly import spherical_polygon


def test_mask_sample():
    """ """
    vertices = [[0,0],[10,0],[10,10],[0,10]]
    S = spherical_polygon(vertices)

    M = Mask(polys=[S], fullsky=False)

    x,y = M.sample(100)

    assert len(x) == 1000
    assert len(y) == 1000
    
    assert np.abs(x.min()) < 1
    assert np.abs(y.min()) < 1
    assert np.abs(x.max() - 10) < 1
    assert np.abs(y.max() - 10) < 1

    r = M.contains(x, y)
    assert np.sum(r) == 0