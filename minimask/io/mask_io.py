



class MaskIO(object):
    """ """
    registry = {
        'mosaic': mosaic.Mosaic,
        'manglepoly': mangle.ManglePolyIO,
    }

    def read(self, filename, format):
        """ """
        if format in self.registry:
            loader = self.registry[format]()
            if loader.canread:
                return loader.read(filename).getmask()

    def write(self, mask, filename, format):
        """ """
        if format in self.registry:
            loader = self.registry[format]()
            if loader.canwrite:
                return loader.write(mask, filename)
