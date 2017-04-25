import os
import numpy as np
import tempfile

import minimask.mask as mask
import minimask.healpix_projection as hp
import minimask.io.mosaic as mosaic


def write_file(filename='test_file_mangle.txt', nside=8):

    tile = [[[-0.5, -0.5],[0.5, -0.5],[0.5,0.5],[-0.5,0.5]]]

    grid = hp.HealpixProjector(nside=nside)

    lon, lat = grid.pix2ang(np.arange(grid.npix))

    centers = np.transpose([lon, lat])

    M = mosaic.mosaic_to_mask(tile, centers)

    M.write(filename, format='manglepoly')

    return filename


def test_mangle():
    filename = tempfile.NamedTemporaryFile(delete=False).name+".mask"

    write_file(filename)

    M = mask.Mask()
    M.load(filename, format='manglepoly')

    os.unlink(filename)

