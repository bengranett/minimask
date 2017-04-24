import numpy as np
from spherical_poly import spherical_polygon
from mask import Mask

def vertices_to_mask(polygons, **metadata):
    """ Construct a mask from a list of longitude, latitude vertices representing
    convex polygons.

    A polygon with N (lon, lat) vertices is specified as a two-dim array
    with shape (N,2).  It is necessary that the number of vertices is N>2.

    Parameters
    ----------
    polygons : list
        A list of arrays representing polygons.

    Other parameters
    ----------------
    key-value pairs
        additional metadata to store with the mask (such as weights)

    Raises
    -------
    Not enough vertices! if N<3.
    """
    self.logger.debug("import %i polygons", len(polygons))

    params = {
        'polys': [],
    }

    params.update(kwargs)

    count = 0
    for i, vertices in enumerate(polygons):
        count += 1
        if self.logger.isEnabledFor(logging.DEBUG):
            step = max(1, len(polygons) // 10)
            if not count % step:
                self.logger.debug("count %i: %f %%", count, count * 100. / len(polygons))

        spoly = spherical_polygon(vertices)
        params['polys'].append(spoly)

    self.logger.info("Loaded %i polygons", len(self.polygons))

    return Mask(**params)
