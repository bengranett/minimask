import numpy as np
import pypelid.utils.sphere as sphere
from sklearn.neighbors import KDTree

class Mask:
	""" """
	polygons = []
	cap_cm = []
	centers = []
	costheta = []

	def __init__(self):
		""" """
		pass

	def import_vertices(self, polygons):
		""" Import a mask as a list of longitude, latitude vertices representing
		convex polygons.

		Inputs
		------
		polygons -

		Outputs
		-------
		"""
		for vertices in polygons:
			lon, lat = np.transpose(vertices)
			xyz = np.transpose(sphere.lonlat2xyz(lon, lat))

			center = np.sum(xyz, axis=0)
			norm = np.sqrt(np.sum(center**2))
			center /= norm

			costheta = np.min(np.dot(xyz, center))

			nvert = len(lon)
			if nvert < 3:
				raise Exception("Not enough vertices! (%i)"%nvert)

			ind1 = np.arange(nvert)
			ind2 = (ind1 + 1) % nvert
			poly = np.cross(xyz[ind1], xyz[ind2])

			# Ensure that the orientations are correct
			# The angle between center and caps should be less than 90 deg
			wrong = np.dot(poly, center) < 0
			poly[wrong] *= -1

			cap_cm = np.ones(len(poly))

			self.polygons.append(poly)
			self.cap_cm.append(cap_cm)
			self.centers.append(center)
			self.costheta.append(costheta)

		self.centers = np.array(self.centers)
		# initialize the tree data structure for quick spatial lookups.
		self.lookup_tree = KDTree(self.centers)
		self.search_radius = np.arccos(np.min(self.costheta))

	def write_mangle_file(self, filename):
		""" Write out a file compatable with mangle.  Let's use FITS format?

		Inputs
		------
		filename

		Outputs
		-------
		"""
		raise Exception("Not implemented")

	def cap_contains(self, cap, cm, point):
		""" Check if a point is contained in a single spherical cap.
		Inputs
		------
		cap - x,y,z tuple of cap center
		cm  - Cap polar angle in Mangle notation cm = 1-cos(theta)
		point - x,y,z tuple of point to test
		"""
		cd = 1 - np.dot(points, cap)
		return cd < cm

	def contains(self, lon, lat):
		""" Check if points are inside mask.
		Inputs
		------
		lon - longitude (degr)
		lat - latitude (degr)

		Outputs
		-------
		bool array
		"""
		lon = np.array(lon)
		lat = np.array(lat)

		xyz = np.transpose(sphere.lonlat2xyz(lon, lat))

		match_list = self.lookup_tree.query_radius(xyz, self.search_radius)

		inside = np.zeros(lon.shape, dtype='bool')  # initially set to False

		for i, matches in enumerate(match_list):
			if len(matches)==0: continue
			for poly_i in matches:
				caps = self.polygons[poly_i]
				cm = self.cap_cm[poly_i]

				ncaps = len(caps)
				r = np.all(self.cap_contains(caps[j], cm[j], xyz[i]) for j in range(ncaps))
				if r:
					inside[i] = r  # set to True
					break
		return inside

	def draw_random_position(self, n):
		""" Draw ra and dec pairs uniformly inside the mask.

		Inputs
		------
		n - number of samples

		Outputs
		-------
		ra, dec
		"""
		lon,lat = sphere.sample_sphere(n)
		sel = self.contains(lon,lat)
		return lon[sel],lat[sel]


if __name__=="__main__":
	test()
