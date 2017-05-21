import logging
import numpy as np
import time
import cPickle as pickle
from sklearn.neighbors import KDTree
import healpy

import io.mask_io as mask_io
import sphere
from spherical_poly import spherical_polygon
import healpix_projection as hp
import misc
import weight_watcher

SPHERE_AREA = 4 * np.pi * (180 / np.pi)**2


class Mask(object):
	""" Routines to process polygon masks. """
	logger = logging.getLogger(__name__)

	def __init__(self,
					filename=None,
					polys=[],
					weights=None,
					metadata=None,
					lookup_tree=None,
					survey_cells=[],
					pixel_lookup=None,
					pixel_mask=None,
					pixel_mask_nside=256,
					pixel_mask_order='ring'):
		""" Routines to process polygon masks.

		A partitioning of the polygon mask will be created using a Healpix grid.
		The resolution and ordering of this grid may be given as arguments.

		Parameters
		----------
		polys : SphericalPolygon
			list
		pixel_mask_nside : int
			Healpix resolution n_side for partitioning the polygon mask.
		pixel_mask_order : str
			Healpix ordering for partitioning the polygonmask (ring or nest).

		"""
		self.config = {
			'pixel_mask_nside': pixel_mask_nside,
			'pixel_mask_order': pixel_mask_order
		}

		self.params = {
			'polys': polys,
			'weights': weights,
			'metadata': metadata,
			'pixel_mask': pixel_mask,
			'pixel_lookup': pixel_lookup,
			'lookup_tree': lookup_tree,
			'survey_cells': survey_cells,
		}

		# initialize pixel mask
		self.grid = hp.HealpixProjector(nside=self.config['pixel_mask_nside'], order=self.config['pixel_mask_order'])

		if filename is not None:
			self.load(filename)

	def __len__(self):
		""" Return number of polygons in mask. """
		try:
			return len(self.params['polys'])
		except TypeError:
			return 0

	def __getitem__(self, ind):
		""" """
		return self.params['polys'][ind]

	def append(self, mask):
		""" Append another mask object to this one. """

		npolys = len(self.params['polys'])

		for poly in mask.params['polys']:
			self.params['polys'].append(poly)

		if mask.params['weights'] is not None:
			if self.params['weights'] is None:
				self.params['weights'] = np.ones(npolys)

			self.params['weights'] = np.concatenate([self.params['weights'],
													mask.params['weights']])
		self.uninit()

	def init(self):
		""" """
		self._build_pixel_mask()

	def uninit(self):
		""" """
		self.params['pixel_mask'] = None
		self.params['lookup_tree'] = None

	def _build_lookup_tree(self):
		""" Private function to initialize lookup trees for fast spatial queries.
		"""
		if len(self.params['polys']) == 0:
			return

		centers = []
		costheta = []
		for S in self.params['polys']:
			centers.append(S.center)
			costheta.append(S.costheta)
		centers = np.array(centers)

		self.logger.debug("Building mask lookup tree")
		self.params['lookup_tree'] = KDTree(centers)
		self.params['search_radius'] = np.arccos(np.min(costheta))
		self.logger.debug("Mask search radius: %f", self.params['search_radius'])

	def _build_pixel_mask(self, expand_fact=1):
		""" Private function to initialize partitioning grid using Healpix."""
		if len(self.params['polys']) == 0:
			# full sky
			self.params['pixel_mask'] = np.ones(self.grid.npix, dtype=bool)
		else:
			if self.params['lookup_tree'] is None:
				self._build_lookup_tree()

			self.logger.debug("build pixel mask nside=%i order=%s", self.grid.nside, self.grid.order)

			self.params['pixel_mask'] = np.zeros(self.grid.npix, dtype=bool)
			pix = np.arange(self.grid.npix)
			xyz = np.transpose(self.grid.pix2vec(pix))
			radius = max(expand_fact * self.grid.pixel_size, self.params['search_radius'])
			matches = self.params['lookup_tree'].query_radius(xyz, radius)
			for i, m in enumerate(matches):
				if len(m) > 0:
					self.params['pixel_mask'][i] = True
		self.params['survey_cells'] = np.where(self.params['pixel_mask'] > 0)[0]
		xyz = np.transpose(self.grid.pix2vec(self.params['survey_cells']))
		self.params['pixel_lookup'] = KDTree(xyz)

	def dump(self, filename):
		""" Write the mask data to disk. """
		t0 = time.time()
		pickle.dump((self.config, self.params), file(filename, "w"))
		dt = time.time()-t0
		self.logger.info("Wrote data to file %s.  time=%f", filename, dt)

	def load_dump(self, filename):
		""" Load a file written by dump. """
		self.logger.debug("Loading mask file %s ...", filename)
		t0 = time.time()
		data = pickle.load(file(filename))
		dt = time.time()-t0

		self.config, self.params = data

		self.grid = hp.HealpixProjector(nside=self.config['pixel_mask_nside'], order=self.config['pixel_mask_order'])

		self.logger.info("Loaded data from file %s.  num centers: %i, dt=%f", filename, len(self.params['polys']), dt)

	def load(self, filename, format=None):
		""" Load a mask file.

		Parameters
		----------
		filename : str
			path to file

		Raises
		------
		IOError if file cannot be parsed.
		"""
		try:
			return self.load_dump(filename)
		except:
			pass

		params = mask_io.read(filename, format)
		self.params.update(params)

	def write(self, filename, format=None):
		""" """
		mask_io.write(self, filename, format)

	def contains(self, lon, lat, get_id=False, get_weights=False):
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
		if len(self.params['polys']) == 0:
			if get_id:
				return np.ones(len(lon), dtype=bool), [[0] for i in xrange(len(lon))]
			else:
				return np.ones(len(lon), dtype=bool)

		if self.params['lookup_tree'] is None:
			self._build_lookup_tree()

		lon = np.array(lon)
		lat = np.array(lat)

		# convert lon,lat angles to unit vectors on the sphere
		xyz = np.transpose(sphere.lonlat2xyz(lon, lat))

		# find polygons near to the points
		match_list = self.params['lookup_tree'].query_radius(xyz, self.params['search_radius'])

		# array to store results, whether points are inside or outside
		inside = np.zeros(lon.shape, dtype='bool')  # initially set to False

		poly_ids = [[] for i in xrange(len(lon))]

		for i, matches in enumerate(match_list):
			# loop through points
			if len(matches) == 0:
				continue
			for poly_i in matches:
				# loop through polygons near to the point

				r = self.params['polys'][poly_i].contains(*xyz[i])
				if r:
					inside[i] = r
					poly_ids[i].append(poly_i)
					if not get_id:
						break

		if get_id:
			return inside, poly_ids
		else:
			return inside

	def get_weight(self, lon, lat):
		""" Return weights of every polygon that contains the points

		Parameters
		----------
		lon : float ndarray
			longitude coordinate (degr)
		lat : float ndarray
			latitude coordinate (degr)

		Returns
		-------
		array
		"""
		inside, poly_ids = self.contains(lon, lat, get_id=True)
		if self.params['weights'] is None:
			out = [np.ones(len(ids)) for ids in poly_ids]
		else:
			out = [np.take(self.params['weights'], ids) for ids in poly_ids]

		return inside, out

	def get_combined_weight(self, lon, lat, operation='sum'):
		""" Return the combined weight of all polygons containing a point.

		Parameters
		----------
		lon : float ndarray
			longitude coordinate (degr)
		lat : float ndarray
			latitude coordinate (degr)
		operation : str or callable
			reduction operation to apply

		Returns
		-------
		array
		"""

		inside, weights = self.get_weight(lon, lat)
		return inside, weight_watcher.combine(weights, operation)

	def sample(self, density=None, n=None,
							cell=None, nside=None, order=None,
							min_sample=100, max_loops=10):
		""" Draw longitude and latitude pairs uniformly inside the mask.

		By default the points are drawn from the full sphere.  If a healpix cell
		number (or list of numbers) is given then randoms will be drawn from
		within those cells only.  In this mode both the healpix nside parameter
		and ordering scheme should be given as arguments.

		After drawing randoms the ones that fall outside the polygon mask are
		discarded.

		Parameters
		----------
		density : float
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
		if self.params['pixel_mask'] is None:
			self._build_pixel_mask()

		if cell is None:
			cell = self.params['survey_cells']   # full sky
		else:
			# sample only selected patches defined by a healpix cell
			cell = self.grid.select_cells(cell, nside, order)
			sel = self.params['pixel_mask'][cell] > 0
			cell = cell[sel]

		if len(cell) == 0:
			# if there are no cells return empty arrays
			return np.array([]), np.array([])

		if misc.is_number(cell):
			n_cells = 1
			cell = int(cell)
		else:
			n_cells = len(cell)

		density_mode = False

		if density is not None:
			density_mode = True
			n = int(SPHERE_AREA * 1. / self.grid.npix * n_cells * density)

		lon_out = []
		lat_out = []

		count = 0
		loop = 0
		while count < n:
			remaining = max(min_sample, n - count)
			lon, lat = self.grid.random_sample(cell, remaining)

			sel = self.contains(lon, lat)
			lon, lat = lon[sel], lat[sel]
			count += len(lon)

			lon_out.append(lon)
			lat_out.append(lat)

			if density is not None:
				break

			if loop > max_loops:
				raise Exception("sample hit max loops! %i"%max_loops)

		lon_out = np.concatenate(lon_out)
		lat_out = np.concatenate(lat_out)

		if not density_mode:
			lon_out = lon_out[:n]
			lat_out = lat_out[:n]

		return lon_out, lat_out

	draw_random_position = sample

	def render(self, res=0):
		""" Generate points along the polygon edges for plotting purposes.

		Parameters
		----------
		res : float
			resolution in degrees used to sample the polygon edges.
			If res=0 only the vertices are returned.

		Returns
		-------
		lists of vertices along edges
		"""
		points = []
		for poly in self.params['polys']:
			points.append(poly.render(res))
		return points

	def pixelize(self, nside=512, order=hp.RING, n=10, weight=True, operation='sum'):
		""" Pixelize the mask using healpix.

		Each healpix cell is sampled randomly with n points.
		The fraction of points inside the mask will be returned
		as a healpix map.

		Parameters
		----------
		nside : int
			healpix nside parameter
		order : str
			healpix order parameter (ring or nest)
		n : int
			number of points to sample in each healpix cell

		Returns
		-------
		numpy.ndarray : healpix map containing area fraction
		"""
		if self.params['pixel_mask'] is None:
			self._build_pixel_mask()

		pixel_mask = healpy.ud_grade(self.params['pixel_mask'].astype('d'),
						order_in=self.grid.order,
						nside_out=nside,
						order_out=order,
						)

		if np.sum(pixel_mask) == 0:
			logging.warning("degraded pixel mask is all zero")

		grid = hp.HealpixProjector(nside=nside, order=order)

		out = np.zeros(grid.npix, dtype='d')

		cells = np.where(pixel_mask > 0)[0]

		for pixel in cells:

			ra, dec = grid.random_sample(pixel, n)

			if weight:
				sel, w = self.get_combined_weight(ra, dec)

				w = w[sel]
				if len(w) == 0:
					continue
				out[pixel] = np.mean(w)
			else:
				sel = self.contains(ra, dec)
				ra = ra[sel]
				if len(ra) == 0:
					continue
				out[pixel] = len(ra) * 1./ n

		return out

	def get_polyid_in_cap(self, lon, lat, theta):
		""" Return a list of polygon ids with centers inside the cap

		Parameters
		----------
		lon : float
			cap center longitude coordinate
		lat : float
			cap center latitude coordinate
		theta : float
			cap opening angle (degree)

		Returns
		-------
		indices : list of polygon indices
		"""
		if self.params['lookup_tree'] is None:
			self._build_lookup_tree()

		xyz = sphere.lonlat2xyz(lon, lat)
		r = np.pi / 180 * theta

		matches = self.params['lookup_tree'].query_radius([xyz], r)[0]

		return matches
