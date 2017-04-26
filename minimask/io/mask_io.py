import os
import logging

import mosaic
import mangle


registry = {
    'mosaic': mosaic.Mosaic,
    'manglepoly': mangle.ManglePolyIO,
}


def read(filename, format=None):
    """ """
    if not os.path.exists(filename):
        raise IOError("File does not exist: '%s'"%filename)

    if format is None:
        for format in registry.keys():
            loader = registry[format]()
            if not loader.canread:
                continue

            try:
                file_is_good = loader.check_file(filename)
            except:
                continue

            if file_is_good:
                return loader.read(filename)
        raise IOError("Cannot find parser that understands the file '%s'"%filename)

    if format in registry:
        loader = registry[format]()
        if loader.canread:
            return loader.read(filename)
        else:
            raise IOError("Parser for format '%s' does not have read functionality."%format)

    raise IOError("Unknown file format: '%s'."%format)



def write(mask, filename, format):
    """ """
    if not format in registry:
        raise IOError("Unknown file format: '%s'."%format)

    if format in registry:
        loader = registry[format]()
        if loader.canwrite:
            return loader.write(mask, filename)
        raise IOError("Parser for format '%s' does not have write functionality."%format)