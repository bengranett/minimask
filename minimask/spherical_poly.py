import logging
import numpy as np
import sphere

DEG2RAD = np.pi / 180


class spherical_polygon(object):
    """ Define a convex spherical polygon as a set of intersecting spherical caps.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, vertices=None):
        """ """
        self.cap_cm = []
        self.caps = []
        self.center = []
        self.costheta = None
        self.ncaps = 0

        if vertices is not None:
            self.vertices_to_caps(vertices)

    def add_cap(self, center, theta, lonlat=False):
        """ Add a cap

        Notes
        -----
        If a polygon is constructed from more than one cap, care must be taken
        that they are loaded in order such that the edges go around counter-
        clockwise.

        Parameters
        ----------
        center : array
            center if spherical cap (either lon,lat or x,y,z)
        costheta : float
            cosine of cap opening angle
        lonlat : bool
            if True interpret center as lon,lat otherwise x,y,z
        """
        if lonlat:
            c = np.array(sphere.lonlat2xyz(*center))
        else:
            c = np.array(center)

        costheta = np.cos(theta * DEG2RAD)

        if self.ncaps > 0:
            self.caps = np.vstack([self.caps, c])
            self.cap_cm = np.concatenate([self.cap_cm, [(1 - costheta)]])
            self.ncaps += 1
            self.compute_center()
        else:
            self.costheta = costheta
            self.caps = np.array([c])
            self.cap_cm = np.array([1 - costheta])
            self.center = c
            self.ncaps += 1

    def compute_center(self, vertices=None):
        """ Compute a pseudo-center of the spherical polygon and compute bounding cap.

        Parameters
        ----------
        vertices : ndarray
            vertices, if not given they will be computed.
        """
        if vertices is None:
            vertices = np.array(sphere.lonlat2xyz(*np.array(self.get_vertices()).T)).T

        center = np.sum(vertices, axis=0)
        norm = np.sqrt(np.sum(center**2))
        self.center = center * 1. / norm
        self.costheta = np.min(np.dot(vertices, self.center))

    def vertices_to_caps(self, vertices):
        """ Convert a polygon defined by lon,lat vertices on a sphere
        to spherical cap format.

        Parameters
        ----------
        vertices : ndarray
        """
        lon, lat = np.transpose(vertices)
        xyz = np.transpose(sphere.lonlat2xyz(lon, lat))

        self.compute_center(xyz)

        nvert = len(lon)
        if nvert < 3:
            raise Exception("Not enough vertices! (%i)" % nvert)

        ind1 = np.arange(nvert)
        ind2 = (ind1 + 1) % nvert
        self.caps = np.cross(xyz[ind2], xyz[ind1])
        norm = np.sum(self.caps * self.caps, axis=1)**.5
        self.caps = np.transpose(self.caps.T / norm)
        # Ensure that the orientations are correct
        # The angle between center and caps should be less than 90 deg
        wrong = np.dot(self.caps, self.center) < 0
        self.caps[wrong] *= -1

        self.cap_cm = np.ones(len(self.caps))

        self.ncaps = len(self.caps)

    def get_vertices(self):
        """ Compute the vertices of the spherical polygon.
        """
        vertices = []
        if self.ncaps == 1:
            return None

        for i in range(self.ncaps):
            j = (i + 1) % self.ncaps

            try:
                p1, p2 = sphere.cap_intersection(self.caps[i], 1 - self.cap_cm[i],
                                                self.caps[j], 1 - self.cap_cm[j])
                vertices.append(p1)

            except sphere.NoIntersection:
                raise

        return vertices

    def rotate(self, lon, lat, orientation=0):
        """ Rotate the spherical polygon.

        A point at (lon,lat) = (0,0) will end up at (lon,lat).

        Parameters
        ----------
        lon : float
            longitudinal angle
        lat : float
            latitudinal angle
        orientation :
            orientation angle from north
        """
        angle = [(orientation, lon, lat)]

        self.center = sphere.rotate_xyz(*self.center, angles=angle)
        self.caps = sphere.rotate_xyz(*self.caps.T, angles=angle).T

    def scale(self, a):
        """ Scale the spherical polygon.

        Parameters
        ----------
        a : float
            scale factor
        """
        boundtheta = np.arccos(self.costheta)
        grow = boundtheta * (a - 1)

        cap_theta = np.arccos(1 - self.cap_cm)

        cap_theta *= 1 + grow / cap_theta

        self.cap_cm = 1 - np.cos(cap_theta)

        self.compute_center()

    def contains(self, lon, lat, z=None, tol=1e-10):
        """ Check if the given coordinates are inside the polygon.

        Parameters
        ----------
        lon : array
            longitude
        lat : array
            latitude
        tol : float
            tolerance

        Returns
        -------
        bool array : True means inside.
        """
        if z is None:
            xyz = np.transpose(sphere.lonlat2xyz(lon, lat))
        else:
            xyz = np.transpose([lon, lat, z])

        if len(xyz.shape) == 1:
            inside = True
        else:
            inside = np.ones(xyz.shape[1])
        for j in range(self.ncaps):
            inside = np.logical_and(inside, (1 - np.dot(xyz, self.caps[j])) < self.cap_cm[j] * (1 + tol))
        return inside

    def render(self, res=10):
        """ Generate lon,lat points along the polygon edges for plotting purposes.

        Parameters
        ----------
        res : float
            step size in degrees to sample the edge.
            res=0 will return only the vertices without points along the edge.
        """
        if self.ncaps == 0:
            return None

        if self.ncaps == 1:
            cap = sphere.render_cap(self.caps[0], 1 - self.cap_cm[0], res=res)
            return cap

        vert = self.get_vertices()

        if res == 0:
            return np.transpose(vert)

        assert(res > 0)

        caps = []
        for i in range(self.ncaps):
            j = (i + 1) % self.ncaps
            a = np.array(sphere.lonlat2xyz(*vert[i]))
            b = np.array(sphere.lonlat2xyz(*vert[j]))
            points = np.vstack([a, b])

            # rotate so cap is at pole
            clon, clat = sphere.xyz2lonlat(*self.caps[j])
            a, b = sphere.rotate_xyz(*points.T, angles=[(0, clon, -90 + clat)], inverse=True).T

            phi_0 = np.arctan2(a[1],a[0])*180/np.pi
            phi_1 = np.arctan2(b[1],b[0])*180/np.pi

            cap = sphere.render_cap(self.caps[j], 1 - self.cap_cm[j], phi_limits=[phi_0, phi_1], res=res)
            caps.append(cap)

        caps = np.hstack(caps)

        return caps
