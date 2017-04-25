import mosaic
import mangle


registry = {
    'mosaic': mosaic.Mosaic,
    'manglepoly': mangle.ManglePolyIO,
}


def read(filename, format):
    """ """
    if format in registry:
        loader = registry[format]()
        if loader.canread:
            return loader.read(filename)


def write(mask, filename, format):
    """ """
    if format in registry:
        loader = registry[format]()
        if loader.canwrite:
            return loader.write(mask, filename)
