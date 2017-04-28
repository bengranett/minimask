import numpy as np
import healpy
import minimask.healpix_projection as hp

def test():
    """ Create healpix projector instance and random sample a cell."""
    H = hp.HealpixProjector(nside=2)

    for n in [0, 100]:

        x, y = H.random_sample(1, n=n)

        assert(len(x) == n)
        assert(len(y) == n)


def check(a, b):
    assert np.allclose(a, b)


def test_select():
    """ test select cell """

    H = hp.HealpixProjector(nside=64, order='ring')

    cases = [
        (16, 'ring', (0)),
        (16, 'ring', (100)),
        (16, 'ring', (100, 101)),
        (2, 'ring', (0)),
        (2, 'ring', (47)),
        (2, 'ring', (10, 11, 12)),
        (16, 'nest', (0)),
        (16, 'nest', (100)),
        (16, 'nest', (100, 101)),
    ]

    for nside, order, coarse_cell in cases:
        cells = H.select_cells(coarse_cell, coarse_nside=nside, coarse_order=order)

        H_low = hp.HealpixProjector(nside=nside, order=order)
        out = H_low.ang2pix(*H.pix2ang(cells))

        out = np.unique(out)

        coarse_cell = np.array([coarse_cell]).flatten()
        coarse_cell.sort()

        out.sort()

        yield check, coarse_cell, out

    error = False
    try:
        cells = H.select_cells(coarse_cell, coarse_nside=64)
    except ValueError:
        error = True

    yield check, error, True
