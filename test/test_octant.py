import os
import numpy as np
import tempfile

import minimask.mask as mask


def check_close(a, b, atol=1e-2):
    assert np.allclose(a, b, atol=atol)

def write_file(filename):
    data = """1 polygons
polygon 0 (3 caps):
0 0 1 1
1 0 0 1
0 1 0 1
"""
    out = file(filename, 'w')
    out.write(data)
    out.close()

    return filename

def test():

    filename = tempfile.NamedTemporaryFile(delete=False).name+".mask"
    write_file(filename)

    M = mask.Mask(filename)

    x, y = M.sample(1)

    ext = [x.min(),x.max(),y.min()]

    yield check_close, len(x), 4*np.pi*(180/np.pi)**2/8., 200

    yield check_close, ext, [0,90,0], 0.1
    yield check_close, y.max(), 90, 2.

    os.unlink(filename)

