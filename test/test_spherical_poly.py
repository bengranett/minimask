import numpy as np
import minimask.spherical_poly as spherical_poly
import minimask.sphere as sphere


def test_equalateral(edge=20.):
    """ construct an equalatoral triangle, rotate it 120 deg and check that
    vertices are the same.
    """
    e = edge
    assert e < 90
    assert e > 0

    cose = np.cos(e * np.pi/180)
    sine = np.sin(e * np.pi/180)

    a = (cose - cose**2) / sine**2
    d = 180. / 2. / np.pi * np.arccos(a)

    poly = [(-d, 90 - e), (d, 90 - e), (0, 90)]

    xyz = sphere.lonlat2xyz(*np.transpose(poly))
    a, b, c = np.transpose(xyz)

    assert np.allclose(np.dot(a, b), np.dot(b, c))
    assert np.allclose(np.dot(a, c), np.dot(b, c))

    center = sphere.xyz2lonlat(*np.sum(xyz, axis=1), getr=True)

    S = spherical_poly.spherical_polygon(poly)

    S.rotate(0, -center[1])

    a = S.get_vertices()

    S.rotate(0, 0, 120)

    b = S.get_vertices()

    assert np.allclose(a[0],b[1])
    assert np.allclose(a[1],b[2])
    assert np.allclose(a[2],b[0])


