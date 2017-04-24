import numpy as np


def mosaic_to_mask(*args, **kwargs):
    """ """
    return Mosaic(*args, **kwargs).generate_mask()


class Mosaic(object):
    """ """
    name = 'mosaic'
    canread = True
    canwrite = False

    def __init__(self, tile=None, centers=None, orientations=None, sizes=None, weights=None):
        """ Construct a mask from a tile that is replicated on the sky
        to form a mosaic.

        Parameters
        ----------
        tile : list
            list of tile patterns in vertex format
        centers : list
            list of centers
        orientations : list
            list of orientations
        scales : list
            list of scale factors
        weights : list
            list of weights
        """

        self.params = {
                    'tile': tile,
                    'centers': centers,
                    'orientations': orientations,
                    'sizes': sizes,
                    'weights': weights
                    }     

    def generate_mask(self):
        """ Construct a mask from a tile that is replicated on the sky
        to form a mosaic.

        Parameters
        ----------
        tile : list
            list of tile patterns in vertex format
        centers : list
            list of centers
        orientations : list
            list of orientations
        scales : list
            list of scale factors
        weights : list
            list of weights
        """

        mask_params = {
            'polys': [],
            'weights': None,
        }

        poly_list = []
        for vertices in tile:
            poly_list.append(spherical_polygon(self.params['vertices']))

        for tile_id, center in enumerate(self.params['centers']):

            for spoly in poly_list:
                new_poly = deepcopy(spoly)
                if self.params['sizes'] is not None and self.params['sizes'][tile_id] != 1:
                    new_poly.scale(self.params['sizes'][tile_id])

                if self.params['orientations'] is not None and self.params['orientations'][tile_id] != 0:
                    new_poly.rotate(*center, orientation=self.params['orientations'][tile_id])

                params['polys'].append(new_poly)
                if params['weights'] is not None:
                    mask_params['weights'].append(params['weights'][tile_id])

        return Mask(**mask_params)

    def write(self, filename):
        """ Write the mask data in mosaic format. """

        ntiles = len(self.params['centers'])

        out = StringIO.StringIO()

        for poly in tile:
            print >>out, "poly", " ".join(["%f"%v for v in poly.flatten()])
        print >>out, "centers"
        for i in range(ntiles):
            print >>out, self.params['centers'][i][0], self.params['centers'][i][1],

            if self.params['orientations'] is None:
                print 0,
            else:
                print >>out, self.params['orientations'][i],

            if self.params['sizes'] is None:
                print 1,
            else:
                print >>out, self.params['sizes'][i],

            if self.params['weights'] is None:
                print 1,
            else:
                print >>out, self.params['weights'][i],

            print "" # end of line

        hash = hashlib.md5(out.getvalue()).hexdigest()

        if filename.endswith("gz"):
            outfile = gzip.open(filename, 'w')
        else:
            outfile = file(filename, 'w')

        outfile.write("# md5sum: {}\n".format(hash))
        outfile.write(out.getvalue())

        out.close()

        self.logger.info("Wrote {} polygons in tile and {} pointing centers from file {}".format(len(tile),len(centers),filename))

    def read(self, filename):
        """ Load a mosaic format file. """
        tile = []
        centers = []

        t0 = time.time()

        try:
            gzip.open(filename).readline()
            input = gzip.open(filename)
        except IOError:
            input = file(filename)

        for line in input:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line.startswith("poly"):
                x = [float(v) for v in line[5:].split()]
                n = len(x) // 2
                tile.append(np.reshape(x, (n, 2)))
            else:
                try:
                    x = [float(v) for v in line.split()]
                except:
                    continue
                center = x.pop(0), x.pop(0)
                try:
                    o = x.pop(0)
                except:
                    o = 0
                try:
                    s = x.pop(0)
                except:
                    s = 1
                try:
                    w = x.pop(0)
                except:
                    w = 1

                centers.append(center)
                orientations.append(o)
                sizes.append(s)
                weights.append(w)

        input.close()

        self.logger.info("Loaded {} polygons in tile and {} pointing centers from file {} (file read time: {:3.1f}sec)".format(len(tile),len(centers),filename, time.time()-t0))
        self.import_mosaic(tile, centers, orientations=orientations, sizes=sizes, weights=weights)
