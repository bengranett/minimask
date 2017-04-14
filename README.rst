minimask
=======================

Light-weight routines for processing sky survey masks

Philosophy 
---------- 

A survey coverage mask is made up of a large number of tiles forming a mosaic
on the sky.  Each tile can be represented as a group of convex spherical
polygons.  We will optimize for the case when the polygons are small, less
than 1 degree, and the mosaic covers a large fraction of the sky. So we can
reference the polygons efficiently by a center point and sort them with
spatial indices.

The main operations to be carried out are:

* query polygons that contain a given point
* draw random samples of points from the area covered

Examples
--------

Dependencies
------------
* `python 2.7 <https://python.org>`_
* `numpy <https://numpy.org>`_
* `scipy <https://scipy.org>`_ 
* `scikit-learn <https://scikit-learn.org>`_
* `healpy <https://github.com/healpy/healpy>`_

Contributors
------------
Ben Granett, Dida Markovic
