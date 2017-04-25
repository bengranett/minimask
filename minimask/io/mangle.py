import numpy as np

from ..spherical_poly import spherical_polygon


def ManglePolyIO(MaskIO):
    """ """
    name = 'mangle_poly'
    canread = True
    canwrite = True

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
            out.write("%i polygons\n" % len(mask.polygons))
            for num in xrange(len(mask.polygons)):
                poly = mask.polygons[num]
                cm = mask.cap_cm[num]
                ncaps = len(poly)
                out.write("polygon %i ( %i caps, 1 weight, 0 pixel, 0 str):\n" % (num, ncaps))
                for i in range(ncaps):
                    x, y, z = np.transpose(poly[i])
                    out.write("%3.15f %3.15f %3.15f %1.10f\n" % (x, y, z, cm[i]))
        self.logger.info("Wrote %i polygons to %s", len(mask.polygons), filename)

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
            words = line.split()

            if words[0] == "polygon":
                weight = 1
                try:
                    weight = float(words[2])
                except:
                    continue
                weights.append(weight)

                num += 1
                if poly is not None:
                    if len(poly) == 0:
                        self.logger.warning("Loading %s (line %i): polygon %i has no caps.",
                            filename, line_num, num)
                        continue
                    polygons.append(poly)
                poly = spherical_polygon()
                continue
            x, y, z, c = [float(v) for v in w]
            poly.add_cap(center=(x,y,z), theta=np.arccos(1-c)*180/np.pi)
        if poly is None:
            self.logger.warning("Failed loading %s: no polygons found!" % filename)
        polygons.append(poly)
        self.logger.info("Loaded %s and got %i polygons", filename, len(polygons))

        params = {'poly': polygons}

        return params
