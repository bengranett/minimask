import numpy as np
import pypelid.utils.misc as misc
import healpy

class HealpixProjector:
	twothirds = 2./3
	deg2rad = np.pi/180.
	rad2deg = 180./np.pi
	sphere_area = 4*np.pi

	def __init__(self, nside=64, order='ring'):
		""" Compute the healpix projection.

		Inputs
		------
		nside - int healpix resolution parameter
		order - string healpix ordering scheme "ring" or "nest"
		"""
		self.nside = nside
		self.npix = healpy.nside2npix(self.nside)
		self.order = order.lower()

		self.nest = False
		if self.order.startswith("n"):
			self.nest = True

		self.pixel_size = np.sqrt(self.sphere_area/self.npix)

	def ang2pix(self, lon, lat):
		""" Wrapper for ang2pix
		Inputs
		------
		lon - longitude (degr)
		lat - latitude (degr)

		Outputs
		-------
		pixel number
		"""
		phi = np.array(lon) * self.deg2rad
		theta = (90 - np.array(lat)) * self.deg2rad
		return healpy.ang2pix(self.nside, theta, phi, self.nest)

	def pix2ang(self, pix):
		""" Wrapper for pix2ang
		Inputs
		------
		pix - int pixel number

		Outputs
		-------
		lon, lat - (degree)
		"""
		theta,phi = healpy.pix2ang(self.nside, pix, self.nest)
		lon = phi * self.rad2deg
		lat = 90 - theta * self.rad2deg
		return lon,lat

	def pix2vec(self, pix):
		""" wrapper for pix2vec """
		return healpy.pix2vec(self.nside, pix, self.nest)

	def vec2pix(self, x, y, z):
		""" wrapper for vec2pix """
		return healpy.vec2pix(self.nside, x, y, z, self.nest)

	def _phitheta2xy(self, phi, theta):
		""" Project points phi,theta onto the healpix plane.

		The projection is described in Gorski et al (2005) and
		Calabretta & Roukema (2007).

		Inputs
		------
		phi - longitude angle (radians [0,2pi]))
		theta - polar angle (radians [0,pi])

		Outputs
		-------
		x,y
		"""
		z = np.cos(theta)

		pole = np.abs(z) > self.twothirds
		equ = np.logical_not(pole)

		x = np.zeros(phi.shape)
		y = np.zeros(phi.shape)

		x[equ] = phi[equ]
		y[equ] = 3./8. * np.pi * z[equ]

		sig = 2 - np.sqrt( 3 * ( 1-np.abs(z[pole]) ) )

		y[pole] = np.pi/4. * sig
		south = (z < 0) & pole
		y[south] *= -1

		phi_t = phi[pole] % (np.pi/2)
		x[pole] = phi[pole] - (np.abs(sig) - 1)*(phi_t - np.pi/4)

		xt = x - y
		yt = x + y

		return xt,yt

	def _xy2phitheta(self, xt, yt):
		""" Inverse projection: compute phi, theta from x,y on the Healpix plane.

		Inputs
		------
		x - x projected coordinate
		y - y projected coordinate

		Outputs
		-------
		phi,theta (radians)
		"""
		x = 0.5*(xt + yt)
		y = 0.5*(-xt + yt)

		equ = np.abs(y) < np.pi/4
		pole = np.logical_not(equ)

		phi = np.zeros(xt.shape)
		z = np.zeros(yt.shape)

		phi[equ] = x[equ]
		z[equ] = 8/3./np.pi*y[equ]

		xt = x[pole] % (np.pi/2)
		a = np.abs(y[pole])-np.pi/4
		b = np.abs(y[pole])-np.pi/2
		phi[pole] = x[pole] - a/b*(xt - np.pi/4)

		z[pole] = (1 - 1./3.*(2-4*np.abs(y[pole])/np.pi)**2)*y[pole]/np.abs(y[pole])

		theta = np.arccos(z)

		return phi,theta

	def pixel_boundaries(self):
		""" """
		raise Exception("not implemented")

	def random_sample(self, pixels, n=1e5):
		""" Sample a healpixel

		Inputs
		------
		pixels - pixel number of list of pixel numbers
		n      - Total number of randoms to draw

		Outputs
		-------
		lon, lat - positions of randoms (degr)
		"""
		if misc.is_number(pixels):
			pixels = [int(pixels)]

		# select pixels to sample
		if len(pixels)==1:
			pix_i = np.zeros(n, dtype=int)
		else:
			pix_i = np.random.choice(len(pixels), n)

		# compute pixel centers
		theta,phi = healpy.pix2ang(self.nside, pixels, nest=self.nest)

		# convert to healpix projection
		xc,yc = self._phitheta2xy(phi, theta)

		# this is the size of a healpix cell in the projection
		step = np.pi/2./self.nside

		# generate randoms in a square
		x,y = np.random.uniform(-0.5,0.5,(2,n))*step

		x += xc[pix_i]
		y += yc[pix_i]

		phi_out, theta_out = self._xy2phitheta(x,y)

		lon = self.rad2deg * phi_out
		lat = 90 - self.rad2deg*theta_out

		return lon,lat
