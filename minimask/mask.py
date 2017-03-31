import logging
import numpy as np
import time
import cPickle as pickle
from sklearn.neighbors import KDTree

import sphere
import healpix_projection as hp
import misc

SPHERE_AREA = 4 * np.pi * (180 / np.pi)**2
DEG2RAD = np.pi / 180


class Mask:
	""" Routines to process polygon masks. """
	logger = logging.getLogger(__name__)

	def __init__(self, pixel_mask_nside=256, pixel_mask_order='ring'):
		""" Routines to process polygon masks.

		A partitioning of the polygon mask will be created using a Healpix grid.
		The resolution and ordering of this grid may be given as arguments.

		Parameters
		----------
		pixel_mask_nside : int
			Healpix resolution n_side for partitioning the polygon mask.
		pixel_mask_order : str
			Healpix ordering for partitioning the polygonmask (ring or nest).

		"""
		self.config = {
			'pixel_mask_nside': pixel_mask_nside,
			'pixel_mask_order': pixel_mask_order
		}

		self.polygons = []
		self.cap_cm = []
		self.centers = []
		self.costheta = []
		self.vertices = []
		self.lookup_tree = None
		self.fullsky = True

		# initialize pixel mask
		self.grid = hp.HealpixProjector(nside=self.config['pixel_mask_nside'], order=self.config['pixel_mask_order'])
		self.pixel_mask = None

	def import_vertices(self, polygons):
		""" Import a mask as a list of longitude, latitude vertices representing
		convex polygons.

		A polygon with N (lon, lat) vertices is specified as a two-dim array
		with shape (N,2).  It is necessary that the number of vertices is N>2.

		Parameters
		----------
		polygons : list
			A list of arrays representing polygons.

		Raises
		-------
		Not enough vertices! if N<3.
		"""
		self.logger.debug("import %i polygons", len(polygons))
		count = 0
		for vertices in polygons:
			count += 1
			if self.logger.isEnabledFor(logging.DEBUG):
				step = max(1, len(polygons) // 10)
				if not count % step:
					self.logger.debug("count %i: %f %%", count, count * 100. / len(polygons))

			lon, lat = np.transpose(vertices)

			xyz = np.transpose(sphere.lonlat2xyz(lon, lat))

			center = np.sum(xyz, axis=0)
			norm = np.sqrt(np.sum(center**2))
			center /= norm

			costheta = np.min(np.dot(xyz, center))

			nvert = len(lon)
			if nvert < 3:
				raise Exception("Not enough vertices! (%i)" % nvert)

			ind1 = np.arange(nvert)
			ind2 = (ind1 + 1) % nvert
			poly = np.cross(xyz[ind2], xyz[ind1])
			norm = np.sum(poly * poly, axis=1)**.5
			poly = np.transpose(poly.T / norm)
			# Ensure that the orientations are correct
			# The angle between center and caps should be less than 90 deg
			wrong = np.dot(poly, center) < 0
			poly[wrong] *= -1

			cap_cm = np.ones(len(poly))

			self.polygons.append(poly)
			self.cap_cm.append(cap_cm)
			self.centers.append(center)
			self.costheta.append(costheta)
			self.vertices.append(vertices)
		self.fullsky = False
		self.logger.info("Loaded %i polygons", len(self.polygons))

	def _build_lookup_tree(self):
		""" Private function to initialize lookup trees for fast spatial queries.
		"""
		if self.fullsky:
			return
		self.logger.debug("Building mask lookup tree")
		self.lookup_tree = KDTree(self.centers)
		self.search_radius = np.arccos(np.min(self.costheta))
		self.logger.debug("Mask search radius: %f", self.search_radius)

	def _build_pixel_mask(self, expand_fact=1):
		""" Private function to initialize partitioning grid using Healpix."""
		if self.fullsky:
			# full sky
			self.pixel_mask = np.ones(self.grid.npix, dtype=bool)
		else:
			if self.lookup_tree is None:
				self._build_lookup_tree()

			self.logger.debug("pixel mask nside=%i order=%s", self.grid.nside, self.grid.order)

			self.pixel_mask = np.zeros(self.grid.npix, dtype=bool)
			pix = np.arange(self.grid.npix)
			xyz = np.transpose(self.grid.pix2vec(pix))
			radius = max(expand_fact * self.grid.pixel_size, self.search_radius)
			matches = self.lookup_tree.query_radius(xyz, radius)
			for i, m in enumerate(matches):
				if len(m) > 0:
					self.pixel_mask[i] = True
		self._cells = np.where(self.pixel_mask > 0)[0]
		xyz = np.transpose(self.grid.pix2vec(self._cells))
		self.pixel_lookup = KDTree(xyz)

	def dump(self, filename):
		""" """
		data = self.polygons, self.cap_cm, self.centers, self.costheta
		t0 = time.time()
		pickle.dump(data, file(filename, "w"))
		dt = time.time()-t0
		self.logger.info("Wrote data to file %s.  time=%f", filename, dt)

	def load(self, filename):
		""" """
		self.logger.debug("Loading mask file %s ...", filename)
		t0 = time.time()
		data = pickle.load(file(filename))
		dt = time.time()-t0
		self.polygons, self.cap_cm, self.centers, self.costheta = data
		if len(self.centers) > 0:
			self.fullsky = False
		self.logger.info("Loaded data from file %s.  num centers: %i, dt=%f", filename, len(self.centers), dt)

	def write_mangle_fits(self, filename):
		""" Write out a mask file in FITS format compatable with mangle.
		Not implemented.

		Parameters
		----------
		filename : str
			Path to output file.
		"""
		raise Exception("Not implemented")

	def write_mangle_poly(self, filename):
		""" Write out a mask file in text format compatable with the Mangle
		polygon format.

		Reference: http://space.mit.edu/~molly/mangle/manual/polygon.html

		Parameters
		----------
		filename : str
			Path to output file.
		"""
		with open(filename, 'w') as out:
			out.write("%i polygons\n" % len(self.polygons))
			for num in xrange(len(self.polygons)):
				poly = self.polygons[num]
				cm = self.cap_cm[num]
				ncaps = len(poly)
				out.write("polygon %i ( %i caps, 1 weight, 0 pixel, 0 str):\n" % (num, ncaps))
				for i in range(ncaps):
					x, y, z = np.transpose(poly[i])
					out.write("%3.15f %3.15f %3.15f %1.10f\n" % (x, y, z, cm[i]))
		self.logger.info("Wrote %i polygons to %s", len(self.polygons), filename)

	def read_mangle_poly(self, filename):
		""" Read in a mask file in Mangle polygon format.

		Reference: http://space.mit.edu/~molly/mangle/manual/polygon.html

		Parameters
		----------
		filename : str
			Path to output file.
		"""
		polygons = []
		cap_cm = []
		poly = None
		cm = None
		num = 0
		line_num = 0
		for line in file(filename):
			line_num += 1
			line = line.strip()
			if line.startswith("#"):
				continue
			words = line.split()
			if words[0] == "polygon":
				num += 1
				if poly is not None:
					if len(poly) == 0:
						self.logger.warning("Loading %s (line %i): polygon %i has no caps.",
							filename, line_num, num)
						continue
					polygons.append(poly)
					cap_cm.append(cm)
				poly = []
				cm = []
				continue
			x, y, z, c = [float(v) for v in w]
			poly.append((x, y, z))
			cm.append(c)
		if poly is None:
			self.logger.warning("Failed loading %s: no polygons found!" % filename)
		polygons.append(poly)
		cap_cm.append(cm)
		self.logger.info("Loaded %s and got %i polygons", filename, len(poly))

	def contains(self, lon, lat):
		""" Check if points are inside mask.

		Parameters
		----------
		lon : float ndarray
			longitude coordinate (degr)
		lat : float ndarray
			latitude coordinate (degr)

		Returns
		-------
		bool array
		"""
		if self.fullsky:
			return np.ones(len(lon), dtype=bool)

		if self.lookup_tree is None:
			self._build_lookup_tree()

		lon = np.array(lon)
		lat = np.array(lat)

		# convert lon,lat angles to unit vectors on the sphere
		xyz = np.transpose(sphere.lonlat2xyz(lon, lat))

		# find polygons near to the points
		match_list = self.lookup_tree.query_radius(xyz, self.search_radius)

		# array to store results, whether points are inside or outside
		inside = np.zeros(lon.shape, dtype='bool')  # initially set to False

		for i, matches in enumerate(match_list):
			# loop through points
			if len(matches) == 0:
				continue
			for poly_i in matches:
				# loop through polygons near to the point
				caps = self.polygons[poly_i]
				cm = self.cap_cm[poly_i]

				ncaps = len(caps)

				# vectorizing this proves slower...
				# inside[i] = np.all((1 - np.sum(xyz[i]*caps,axis=1)) < cm)

				# loop through each cap and test if the point is inside.
				# stop after the first fail
				for j in range(ncaps):
					r = (1 - np.dot(xyz[i], caps[j])) < cm[j]
					if not r:
						break
				if r:
					inside[i] = r  # set to True
					break
		return inside

	def _select_cells(self, coarse_cell, coarse_nside, coarse_order):
		""" Private function returns list of cells in the internal healpix map
		that fall in a given patch of sky.

		The resoultion of the internal map must be higher than the coarse resolution
		input here.

		Parameters
		----------
		coarse_cell : int
			cell number or list defining patch of sky
		coarse_nside : int
			nside of pixelization
		coarse_order : str
			pixelization order ('ring' or 'nest')

		Returns
		-------
		list : cell indices in pixel map
		"""
		# If the resolution given is higher than the internal pixel map,
		# rebuild the map at an even higher resolution.
		if coarse_nside > self.grid.nside:
			self.grid = hp.HealpixProjector(nside=2 * coarse_nside, order=coarse_order)
			self._build_pixel_mask()

		# make sure input is iterable
		if misc.is_number(coarse_cell):
			coarse_cell = [int(coarse_cell)]

		coarse_grid = hp.HealpixProjector(nside=coarse_nside, order=coarse_order)
		radius = coarse_grid.pixel_size

		xyz = np.transpose(coarse_grid.pix2vec(coarse_cell))
		matches = self.pixel_lookup.query_radius(xyz, radius)
		matches = np.concatenate(matches)

		xyz = np.take(self.pixel_lookup.data, matches, axis=0)
		pix = coarse_grid.vec2pix(*xyz.transpose())
		ii = pix == coarse_cell
		matches = self._cells[matches[ii]]
		return matches

	def draw_random_position(self, dens=None, n=None,
							cell=None, nside=1, order=hp.RING):
		""" Draw longitude and latitude pairs uniformly inside the mask.

		By default the points are drawn from the full sphere.  If a healpix cell
		number (or list of numbers) is given then randoms will be drawn from
		within those cells only.  In this mode both the healpix nside parameter
		and ordering scheme should be given as arguments.

		After drawing randoms the ones that fall outside the polygon mask are
		discarded.

		Parameters
		----------
		dens : float
			number density of samples (number per square degree)
		cell : int or list
			optional healpix cell number or list of cell numbers
		nside : int
			healpix nside parameter
		nest : bool
			if True use Nest otherwise use Ring ordering

		Returns
		-------
		lon, lat : random coordinates
		"""
		if cell is not None:
			self.logger.debug("selected cell: %s (nside %s)", cell, nside)

		if self.pixel_mask is None:
			self._build_pixel_mask()

		if cell is None:
			cell = self._cells   # full sky
		else:
			# sample only selected patches defined by a healpix cell
			cell = self._select_cells(cell, nside, order)

		if len(cell) == 0:
			# if there are no cells return empty arrays
			# self.logger.warning("Effective area of the survey is 0.")
			return np.array([]), np.array([])

		if misc.is_number(cell):
			n_cells = 1
			cell = int(cell)
		else:
			n_cells = len(cell)

		if dens is not None:
			n = int(SPHERE_AREA * 1. / self.grid.npix * n_cells * dens)

		self.logger.debug("Random sampling: npoints=%i, ncells=%i", n, len(cell))
		t0 = time.time()

		lon, lat = self.grid.random_sample(cell, n)

		sel = self.contains(lon, lat)
		self.logger.debug("done! elapsed time = %f sec", time.time() - t0)
		self.logger.debug("Random sampling success rate: %f", np.sum(sel) * 1. / len(sel))

		return lon[sel], lat[sel]

	sample = draw_random_position