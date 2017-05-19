import logging
import numpy as np
import minimask.healpix_projection as hp
import minimask.sphere as sphere

def run(nside=8, n = 1000000):
    """ test cap that encloses the healpix cell """

    grid = hp.HealpixProjector(nside=nside)

    lon, lat = sphere.sample_sphere(n)

    pix = grid.ang2pix(lon, lat)

    for i in range(grid.npix):

        lon_c, lat_c, theta_c = grid.get_cap(i)

        sel = pix == i

        if np.sum(sel) == 0:
            continue

        d = sphere.distance(lon_c, lat_c, lon[sel], lat[sel])

        logging.debug("nside %i pix %i d.max() %f < %f", nside, i, d.max(), theta_c)

        assert d.max() < theta_c


def test():
    for nside in (1, 2, 4, 8, 16):

        yield run, nside