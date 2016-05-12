"""spherical geometry utilities"""

import numpy as np

# degrees to radian conversions
c = np.pi/180
ic = 180/np.pi

def distance(ra1,dec1,ra2,dec2):
	""" compute distance between two points on the sphere using the haversine formula

	Inputs
	------
	ra1
	dec1
	ra2
	dec2

	Outputs
	-------
	distance (degree)
	"""
	dra = c*(ra1 - ra2)
	ddec = c*(dec1 - dec2)
	r = np.sin(dra/2)**2 + np.cos(dec1*c)*np.cos(dec2*c)*np.sin(ddec/2)**2
	return 2*np.arcsin(np.sqrt(r))*ic

def lonlat2xyz(lon,lat,r=1):
	""" """
	x = r*np.cos(lon*c)*np.cos(lat*c)
	y = r*np.sin(lon*c)*np.cos(lat*c)
	z = r*np.sin(lat*c)
	return x,y,z

def xyz2lonlat(x,y,z,getr=False):
	""" """
	if getr:
		r = np.sqrt(x*x+y*y+z*z)
	else:
		r = np.ones(x.shape)
	lat = np.arcsin(z/r)*ic
	lon = np.arctan2(y,x)*ic
	if getr:
		return lon,lat,r
	return lon,lat

def rotate_xyz(x,y,z,angles=None,inverse=False):
	""" Rotate a set of vectors pointing in the direction x,y,z

	angles is a list of longitude and latitude angles to rotate by.
	First the longitude rotation is applied (about z axis), then the
	latitude angle (about y axis).
	"""
	if angles==None:
		return x,y,z

	xyz = np.array([x,y,z])
	for dphi,dlon,dlat in angles:
		dphi*=c
		dlon*=c
		dlat*=c
		m0 = np.array([[1,0,0],
					  [0, np.cos(dphi),np.sin(dphi)],
					  [0, -np.sin(dphi), np.cos(dphi)]])

		m1 = np.array([[np.cos(dlon),-np.sin(dlon),0],
					  [np.sin(dlon), np.cos(dlon),0],
					  [0,0,1]])

		m2 = np.array([[np.cos(dlat),0,-np.sin(dlat)],
					  [0,1,0],
					  [np.sin(dlat), 0, np.cos(dlat)]])

		m = np.dot(np.dot(m1,m2),m0)

	if inverse:
		m = np.linalg.inv(m)

	xyz2 = np.dot(m,xyz)
	return xyz2

def rotate_lonlat(lon,lat,angles=[(0,0)], inverse=False):
	""" Rotate a set of longitude and latitude coordinate pairs.
	"""
	xyz = np.array(lonlat2xyz(lon,lat))
	xyz2 = rotate_xyz(*xyz,angles=angles, inverse=inverse)
	return xyz2lonlat(*xyz2,getr=False)

def sample_sphere(n):
	""" Sample points on a sphere.
	Inputs
	------
	n - number of points to draw

	Outputs
	-------
	lon, lat - length n arrays (degrees)
	"""
	xyz = np.random.normal(0,1,(3,n))
	norm = np.sqrt(np.sum(xyz**2,axis=0))
	xyz /= norm
	lon, lat = xyz2lonlat(*xyz)
	return lon, lat

def sample_cap(n, lon=None, lat=None, xyz=None, theta=180, costheta=None):
	""" Sample points on a spherical cap.

	Inputs
	------

	Outputs
	-------
	"""
	if costheta is None:
		costheta = np.cos(theta*c)
	if costheta <= -1:
		return sample_sphere(n)

	if lon is None:
		lon,lat = xyz2lonlat(*xyz)

	l = np.random.uniform(0,2*np.pi,n)
	z = np.random.uniform(costheta,1,n)

	x = np.cos(l) * np.sqrt(1-z**2)
	y = np.sin(l) * np.sqrt(1-z**2)

	xyzt = rotate_xyz(x,y,z,angles=[(0,lon,90-lat)])
	return xyz2lonlat(*xyzt)
