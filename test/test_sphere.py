
import sys
import numpy as np
import minimask.sphere as sphere


def check_close(a, b, rtol=1e-5):
    assert np.allclose(a, b, rtol=rtol)

def test_distance():
    cases = [
        ((0,0),(10,0),10),
        ((0,0),(20,0),20),
        ((0,0),(30,0),30),
        ((0,0),(90,0),90),
        ((0,0),(180,0),180),
        ((0,0),(181,0),179),
        ((0,0),(360,0),0),

        ((0,0),(0,10),10),
        ((0,0),(0,45),45),
        ((0,0),(0,90),90),
        ((0,0),(0,91),91),
        ((0,0),(180,89),91),
        ((0,0),(180,0),180),
        ((0,0),(180,-90),90),

        ((0,90),(179,89),1)
        ]

    for a, b, r in cases:
        d = sphere.distance(a[0], a[1], b[0], b[1])
        yield check_close, r, d

def test_lonlat2xyz():
    cases = [
        ((0,0),(1,0,0)),
        ((0,90),(0,0,1)),
    ]
    for ll, xyz in cases:
        xyz2 = sphere.lonlat2xyz(*ll)
        yield check_close, xyz, xyz2

def test_lonlat2xyz_bothways():
    lon,lat = np.meshgrid(np.arange(0,360,10), np.arange(-90,90,10))
    lon = lon.flatten()
    lat = lat.flatten()
    xyz = sphere.lonlat2xyz(lon, lat)
    lon2,lat2 = sphere.xyz2lonlat(*xyz)

    check_close([lon,lat], [lon2%360,lat2])

def test_rotate_xyz():
    cases = [
        ((1,0,0),[(0,10,0)],sphere.lonlat2xyz(10,0)),
        ((1,0,0),[(0,360,0)],(1,0,0)),
        ((1,0,0),[(0,180,0)],(-1,0,0)),
        ((0,0,1),[(90,0,0)],(0,1,0)),
        ((0,0,1),[(0,0,-90)],(1,0,0)),
        ((0,0,1),[(0,0,90)],(-1,0,0))
    ]
    for xyz, angles, res in cases:
        xyz2 = sphere.rotate_xyz(*xyz, angles=angles)
        yield check_close, res, xyz2

        xyz2 = sphere.rotate_xyz(*res, angles=angles, inverse=True)
        yield check_close, xyz, xyz2

def test_sample_sphere(n=1000):
    lon,lat = sphere.sample_sphere(n)
    assert np.all(np.isfinite(lon))    
    assert np.all(np.isfinite(lat))

def test_sample_cap(n=1000):
    cases = [
        ((0,0), 45),
        ((0,0), 2.),
        ((0,90), 1.),
        ((0,10), 10.),
        ((30,70), 30.),
        ((30,70), -1),
    ]
    for center, theta in cases:
        lon,lat = sphere.sample_cap(n, lon=center[0], lat=center[1], theta=theta)
        r = sphere.distance(center[0], center[1], lon, lat)
        yield check_close, r.max(), np.abs(theta), 1e-2
