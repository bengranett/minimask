import numpy as np
import logging

from ..spherical_poly import spherical_polygon


class ManglePolyIO(object):
    """ """
    logger = logging.getLogger(__name__)

    name = 'mangle_poly'
    canread = True
    canwrite = True

    def check_file(self, filename, nbytes=1000):
        """ Check if the file looks like mangle format. """
        self.logger.debug("Checking format...")

        data = file(filename).read(nbytes)

        lc = 0
        for line in data.split("\n"):
            line = line.strip()
            if line == "": continue
            if line.startswith("#"): continue

            lc += 1

            if lc > 3:
                break

            words = line.split()

            if lc == 1:
                if not words[1] == "polygons":
                    return False
                try:
                    int(words[0])
                except ValueError:
                    return False
                continue

            if lc == 2:
                if not words[0] == "polygon":
                    return False
                continue

            if lc == 3:
                if len(words) != 4:
                    return False

        return True

    def write(self, mask, filename):
        """ Write out a mask file in text format compatable with the Mangle
        polygon format.

        Reference: http://space.mit.edu/~molly/mangle/manual/polygon.html

        Parameters
        ----------
        filename : str
            Path to output file.
        """
        with open(filename, 'w') as out:
            out.write("%i polygons\n" % len(mask.params['polys']))
            for num, poly in enumerate(mask.params['polys']):
                out.write("polygon %i ( %i caps, 1 weight, 0 pixel, 0 str):\n" % (num, poly.ncaps))
                for i in range(poly.ncaps):
                    x, y, z = np.transpose(poly.caps[i])
                    out.write("%3.15f %3.15f %3.15f %1.10f\n" % (x, y, z, poly.cap_cm[i]))
        self.logger.info("Wrote %i polygons to %s", len(mask.params['polys']), filename)

    def read(self, filename):
        """ Read in a mask file in Mangle polygon format.

        Reference: http://space.mit.edu/~molly/mangle/manual/polygon.html

        Parameters
        ----------
        filename : str
            Path to output file.
        """
        polygons = []
        weights = []
        poly = None
        num = 0
        line_num = 0
        for line in file(filename):
            line_num += 1
            line = line.strip()
            if line.startswith("#"):
                continue
            line = line.replace(",", " ")
            words = line.split()

            if len(words) < 3:
                continue

            if words[0] == "polygon":
                try:
                    weight = float(words[words.index('weight')-1])
                except:
                    weight = 1
                weights.append(weight)

                num += 1
                if poly is not None:
                    if poly.ncaps == 0:
                        self.logger.warning("Loading %s (line %i): polygon %i has no caps.",
                            filename, line_num, num)
                        continue
                    polygons.append(poly)
                poly = spherical_polygon()
                continue
            x, y, z, c = [float(v) for v in words]
            poly.add_cap(center=(x,y,z), theta=np.arccos(1-c)*180/np.pi)
        if poly is None:
            self.logger.warning("Failed loading %s: no polygons found!" % filename)
        polygons.append(poly)
        self.logger.info("Loaded %s and got %i polygons", filename, len(polygons))

        params = {
            'poly': polygons,
            'weights': weights,
        }

        return params
