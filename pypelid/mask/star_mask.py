import numpy as np
from pypelid import params
from pypelid.utils import sphere
import logging

class StarMask(object):
	""" Bright star mask """

	def __init__(self, config = None):
		""" """
		if config is None:
			config = params.Config()
		self.config = config

	def get_radius(self, mag):
		""" Return obscuration radius given magnitude.

		Inputs
		------
		mag

		Outputs
		-------
		radius - arcsec
		"""
		fit = self.config['star_size_fit']
		radius = 10**(fit[0] * mag + fit[1])
		return radius

	def hit(self, star_catalogue, ra, dec):
		""" Determine if points are near to a star based upon a circular aperture.

		Inputs
		------
		star_catalogue - catalogue object
		ra             - RA can be image or sky coordinates to match catalogue.
		dec            - Dec can be image or sky coordinates to match catalogue.

		Outputs
		-------
		bool array     - True if hits a star, False otherwise
		"""
		logging.debug("querying bright star mask, cat %i, query points %i",len(star_catalogue),len(ra))
		hits = np.zeros(len(ra), dtype=bool)

		if len(ra)==0:
			return hits

		radius = self.get_radius(star_catalogue['mag'])/3600.

		if len(star_catalogue) < len(ra):
			# number of stars is less than number of query points.
			# in this case loop over stars.
			tmp_cat = type(star_catalogue)(lon=ra, lat=dec)
			matches = tmp_cat.query_cap(star_catalogue.lon, star_catalogue.lat, radius=radius)
			for i in range(len(star_catalogue)):
				hits[matches[i]] = True
		else:
			# number of stars is greater than number of query points.
			# loop over query points.
			all_matches = star_catalogue.query_cap(ra, dec, radius.max())
			for i in range(len(ra)):
				matches = all_matches[i]
				if len(matches) == 0: continue
				d = sphere.distance(ra[i],dec[i],
					star_catalogue.lon[matches], star_catalogue.lat[matches])
				if np.any(d < radius[matches]):
					hits[i] = True
		return hits

	__call__ = hit


if __name__ == "__main__":
	SM = StarMask()
	x = np.linspace(12,20)
	y = SM.get_radius(x)
	import pylab
	pylab.plot(x,y)
	pylab.show()
