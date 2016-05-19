import numpy as np
import pypelid.utils.sphere as sphere
import pypelid.utils.misc as misc
import pypelid.vm.healpix_projection as hp
from sklearn.neighbors import KDTree
import healpy
import time

SPHERE_AREA = 4*np.pi*(180/np.pi)**2
DEG2RAD = np.pi/180

class Mask:
	""" """

	def __init__(self, pixel_mask_nside=64, pixel_mask_order='ring'):
		""" """
		self.polygons = []
		self.cap_cm = []
		self.centers = []
		self.costheta = []
		self.vertices = []

		# initialize pixel mask
		self.pixel_mask_nside = pixel_mask_nside
		self.pixel_mask_order = pixel_mask_order
		self.HP = hp.HealpixProjector(self.pixel_mask_nside,self.pixel_mask_order)
		self.pixel_mask = np.zeros(self.HP.npix, dtype=bool)

	def import_vertices(self, polygons):
		""" Import a mask as a list of longitude, latitude vertices representing
		convex polygons.

		Inputs
		------
		polygons -

		Outputs
		-------
		"""
		self._import_vertices(polygons)
		self.build_pixel_mask()


	def _import_vertices(self, polygons):
		""" Import a mask as a list of longitude, latitude vertices representing
		convex polygons.

		Inputs
		------
		polygons -

		Outputs
		-------
		"""
		t0 = time.time()
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
			poly = np.cross(xyz[ind2], xyz[ind1])

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

		print "import vert",time.time()-t0

		self.centers = np.array(self.centers)
		# initialize the tree data structure for quick spatial lookups.
		self.lookup_tree = KDTree(self.centers)
		self.search_radius = np.arccos(np.min(self.costheta))


	def build_pixel_mask(self, expand_fact=1):
		""" """
		pix = np.arange(self.HP.npix)
		xyz = np.transpose(self.HP.pix2vec(pix))
		radius = expand_fact * self.HP.pixel_size
		matches = self.lookup_tree.query_radius(xyz, radius)
		for i,m in enumerate(matches):
			if len(m)>0:
				self.pixel_mask[i] = True
		self._cells = np.where(self.pixel_mask > 0)[0]
		xyz = np.transpose(self.HP.pix2vec(self._cells))
		self.pixel_lookup = KDTree(xyz)


	def write_mangle_file(self, filename):
		""" Write out a file compatable with mangle.  Let's use FITS format?

		Inputs
		------
		filename

		Outputs
		-------
		"""
		raise Exception("Not implemented")

	def cap_contains(self, cap, cm, points):
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
				for j in range(ncaps):
					r = (1 - np.dot(xyz[i], caps[j])) < cm[j]
					if not r: break
				if r:
					inside[i] = r  # set to True
					break
		return inside

	def _select_cells_check(self, coarse_cell, coarse_nside, coarse_order):
		""" Returns list of cells in the internal healpix map that fall in a
		given patch of sky.  The resoultion of the internal map must be higher
		than the coarse resolution input here.

		Inputs
		------
		coarse_cell - cell number or list defining patch of sky
		coarse_nside - nside of pixelization
		coarse_order - pixelization order

		Outputs
		-------
		list of cells in pixel map
		"""
		assert(self.pixel_mask_nside > coarse_nside)
		t0 = time.time()
		if misc.is_number(coarse_cell):
			coarse_cell = [int(coarse_cell)]
		print coarse_cell

		clon,clat = self.HP.pix2ang(self._cells)

		# compute pixel numbers on coarser grid
		coarse_grid = hp.HealpixProjector(coarse_nside, coarse_order)
		id_coarse = coarse_grid.ang2pix(clon,clat)

		matches = []
		for p in coarse_cell:
			sel = np.where(id_coarse == p)
			matches.append(self._cells[sel])
		matches = np.concatenate(matches)
		print "select cells",time.time()-t0
		return matches

	def _select_cells(self, coarse_cell, coarse_nside, coarse_order):
		""" Returns list of cells in the internal healpix map that fall in a
		given patch of sky.  The resoultion of the internal map must be higher
		than the coarse resolution input here.

		Inputs
		------
		coarse_cell - cell number or list defining patch of sky
		coarse_nside - nside of pixelization
		coarse_order - pixelization order

		Outputs
		-------
		list of cells in pixel map
		"""
		assert(self.pixel_mask_nside > coarse_nside)
		t0 = time.time()
		if misc.is_number(coarse_cell):
			coarse_cell = [int(coarse_cell)]

		coarse_grid = hp.HealpixProjector(coarse_nside, coarse_order)
		radius = coarse_grid.pixel_size

		xyz = np.transpose(coarse_grid.pix2vec(coarse_cell))
		matches = self.pixel_lookup.query_radius(xyz, radius)
		matches = np.concatenate(matches)

		xyz = np.take(self.pixel_lookup.data,matches,axis=0)
		pix = coarse_grid.vec2pix(*xyz.transpose())
		ii = pix==coarse_cell
		matches = self._cells[matches[ii]]
		print "select cells fast",time.time()-t0
		return matches

	def draw_random_position(self, dens, cell=None, nside=1, order='ring'):
		""" Draw ra and dec pairs uniformly inside the mask.

		By default the points are drawn from the full sphere.  If a healpix cell
		number (or list of numbers) is given then randoms will be drawn from
		within those cells only.  In this mode both the healpix nside parameter
		and ordering scheme should be given as arguments.

		After drawing randoms the ones that fall outside the pointing mask are
		discarded.

		Inputs
		------
		dens - number density of samples (number per square degree)
		cell - optional healpix cell number or list of cell numbers
		nside - healpix nside parameter
		nest - if True use Nest otherwise use Ring ordering

		Outputs
		-------
		ra, dec
		"""
		if cell is None:
			cell = self._cells   # full sky
		else:
			# sample only selected patches defined by a healpix cell
			cell = self._select_cells(cell,nside,order)

		if len(cell)==0:
			# if there are no cells return empty arrays
			return np.array([]),np.array([])

		if misc.is_number(cell):
			n_cells = 1
			cell = int(cell)
		else:
			n_cells = len(cell)

		n = int(SPHERE_AREA * 1. / self.HP.npix * n_cells * dens)

		lon,lat = self.HP.random_sample(cell, n)

		sel = self.contains(lon,lat)
		print "eff",np.sum(sel)*1./len(sel)

		return lon[sel],lat[sel]

	def draw_random_position_slow(self, dens, cell=None, nside=1, order='ring'):
		""" This routine is here for testing purposes.  It samples from the full
		sphere or from a given healpix cell without further optimization."""
		if cell is None:
			n = int(SPHERE_AREA * dens)
			lon,lat = sphere.sample_sphere(n)
			sel = self.contains(lon,lat)
			print "eff",np.sum(sel)*1./len(sel)
			return lon[sel],lat[sel]

		if misc.is_number(cell):
			n_cells = 1
		else:
			n_cells = len(cell)

		HP = hp.HealpixProjector(nside, order)
		n = int(SPHERE_AREA * 1. / HP.npix * n_cells * dens)

		lon,lat = HP.random_sample(cell, n)

		sel = self.contains(lon,lat)

		print "eff",np.sum(sel)*1./len(sel)
		return lon[sel],lat[sel]



if __name__=="__main__":
	test()
