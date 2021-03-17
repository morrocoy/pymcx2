# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Tue Mar 16 7:12:51 2021

@author: kpapke
"""
import numpy as np

from .log import logmanager

__all__ = ['MCHFileChunk']

logger = logmanager.getLogger(__name__)


class MCHFileChunk(object):
    """ Represents a chunk of data in an mch file


    Attributes
    ----------
    position : uint64
        The chunk start position in file.
    next_position : uint64
        The next chunk start position in file.
    next_offset : uint64
        The position of the next chunk relative to this one.
    meta_offset : uint64
        The position of the metadata start within chunk.
    data_offset : uint64
        The position of the data start within chunk.
    seed_offset : uint64
        The position of the seeds start within chunk.
    dtype: data-type
        The type of the statistical data stored in the file.
    """

    def __init__(self, file, offset=0, dtype='<f'):
        """

        Parameters
        ----------
        mchfile : file object
            An already opened mch file.
        offset : uint64, optional
            Offset to the current position in file. Default is 0.
        dtype: data-type, optional
            The type of the statistical data stored in the file. Default is
            float with little endianness.
        """
        self.file = None  # input file
        self.position = None  # chunk start position in file
        self.next_position = None  # next chunk start position in file
        self.next_offset = None  # position of next chunk relative to this one
        self.meta_offset = None  # position of meta data start within chunk
        self.data_offset = None  # position of data start within chunk
        self.seed_offset = None  # position of seeds start within chunk


        # validate input file
        if hasattr(file, "read"):
            position = file.tell()
            file.seek(position + offset)
            logger.debug("Reading chunk header at {}".format(file.tell()))
            buffer = file.read(4)
            if buffer == b'MCXH':
                logger.debug("Found valid chunk header: {}".format(buffer))
                self.file = file
                self.position = position + offset
                self.meta_offset = 4  # position of meta data start within chunk
            else:
                logger.debug("Found invalid chunk header: {}".format(buffer))
                file.seek(position)  # return to the original position
        else:
            raise RuntimeError("Cannot read data from the underlying mch file.")

        # data arrays
        self.dtype = np.dtype(dtype)  # type of statistical data
        self.data = None  # statistical data
        self.seed = None  # seeds

        # attributes of metadata
        self.version = None  # version of the mch file

        self.ncol = None  # number of columns in the raw data array
        self.nmat = None  # number of mediums
        self.ndet = None  # number of detectors
        self.nsrc = None  # number of sources

        self.totalphoton = None  # total number of launched photons
        self.detectedphoton = None  # number of detected photons
        self.savedphoton = None  # number of saved photons

        self.flags = None  # to fill up bytes

        self.lengthunit = None  # length unit to scale partial paths
        self.seedbyte = None  #
        self.normalizer = None
        self.respin = None
        self.junk = None

        # load metadata
        if self.file is not None:
            self.loadMetadata()


    def _unravelFlags(self, flags):
        """Unravel the flags into a readable dictionary.

        Bit MCX Flag  Key       Description
        === ========= ========= ================================
        1   D         detectid  Detector ID (1)
        2   S         nscatter  Scattering event counts (#media)
        3   P         ppathlen  Partial path-lengths (#media)
        4   M         momentum  Momentum transfer (#media)
        5   X         pos_exit  Exit position (3)
        6   V         dir_exit  Exit direction (3)
        7   W         weight    Initial weight (1)
        8   -         bit_8     Not used
        === ========= ========= ================================

        """
        keys = [
            "detectid",
            "nscatter",
            "ppathlen",
            "momentum",
            "pos_exit",
            "dir_exit",
            "weight",
            "bit_8",
        ]
        values = [int(val) for val in list(np.binary_repr(flags % 256, 8))]

        # little or native byte ordering (">", "!")
        if self.dtype.byteorder in ("<", "="):
            return dict(zip(keys, values[::-1]))
        else:  # big byte ordering (">", "!")
            return dict(zip(keys, values))


    def clear(self):
        self.file = None
        self.position = None  # offset for the chunk in mch file
        self.next_position = None
        self.seed_offset = None  # position of the seeds start within chunk

        self.version = None  # version of the mch file

        self.ncol = None  # number of columns in the raw data array
        self.nmat = None  # number of mediums
        self.ndet = None  # number of detectors
        self.nsrc = None  # number of sources

        self.totalphoton = None  # total number of launched photons
        self.detectedphoton = None  # number of detected photons
        self.savedphoton = None  # number of saved photons

        self.flags = None  # to fill up bytes

        self.lengthunit = None  # length unit to scale partial paths
        self.seedbyte = None  #
        self.normalizer = None
        self.respin = None
        self.junk = None


    def empty(self):
        """Check if the chunk is empty."""
        if self.file is None:
            return True
        else:
            return False


    def loadData(self):
        """Read data of the mch file chunk and store them internally.
        """
        self.data = self.readData()

    def loadSeed(self):
        """Read seeds of the mch file chunk and store them internally.
        """
        self.seed = self.readSeed()


    def loadMetadata(self):
        """Read metadata of the mch file chunk. """
        if self.empty():
            return -1

        # skip four bytes header
        self.file.seek(self.position + self.meta_offset)

        # verify file version
        logger.debug("Reading chunk version at {}".format(self.file.tell()))
        version = np.fromfile(self.file, dtype=np.uint32, count=1)[0]
        if version != 1:
            logger.debug("Version higher than 1 is not supported")
            return -2

        # read parameters
        logger.debug("Reading chunk metadata at {}".format(self.file.tell()))
        nmat = np.fromfile(self.file, dtype=np.uint32, count=1)[0]
        ndet = np.fromfile(self.file, dtype=np.uint32, count=1)[0]
        ncol = np.fromfile(self.file, dtype=np.uint32, count=1)[0]
        totalphoton = np.fromfile(self.file, dtype=np.uint32, count=1)[0]
        detectedphoton = np.fromfile(self.file, dtype=np.uint32, count=1)[0]
        savedphoton = np.fromfile(self.file, dtype=np.uint32, count=1)[0]
        unitmm = np.fromfile(self.file, dtype=np.float32, count=1)[0]
        seedbyte = np.fromfile(self.file, dtype=np.uint32, count=1)[0]
        normalizer = np.fromfile(self.file, dtype=np.float32, count=1)[0]
        respin = np.fromfile(self.file, dtype=np.int32, count=1)[0]
        nsrc = np.fromfile(self.file, dtype=np.uint32, count=1)[0]
        flags = np.fromfile(self.file, dtype=np.uint32, count=1)[0]
        junk = np.fromfile(self.file, dtype=np.int32, count=2)

        # update chunk offsets and positions in underlying mch file
        self.data_offset = 64
        self.seed_offset = self.data_offset + 4 * savedphoton * ncol
        self.next_offset =  self.seed_offset + savedphoton * seedbyte
        self.next_position = self.position + self.next_offset

        self.version = version

        self.ncol = ncol
        self.nmat = nmat
        self.ndet = ndet
        self.nsrc = nsrc

        self.totalphoton = totalphoton * respin if respin > 1 else totalphoton
        self.detectedphoton = detectedphoton
        self.savedphoton = savedphoton

        self.flags = self._unravelFlags(flags)

        self.lengthunit = unitmm
        self.seedbyte = seedbyte
        self.normalizer = normalizer
        self.respin = respin
        self.junk = junk

        return 0  # no error


    def readData(self):
        """Read data and seed of the mch file chunk. """
        if self.empty() or self.data_offset is None:
            return None

        # skip header and metadata
        self.file.seek(self.position + self.data_offset)

        # read data
        logger.debug("Reading chunk data at {}".format(self.file.tell()))
        shape = (self.savedphoton, self.ncol)
        buffer = np.fromfile(self.file, dtype=self.dtype, count=np.prod(shape))
        data = buffer.reshape(shape)

        return data


    def readSeed(self):
        """Read data and seed of the mch file chunk. """
        if self.empty() or self.seed_offset is None:
            return None

        # skip header and metadata
        self.file.seek(self.position + self.seed_offset)

        # read seeds
        if self.seedbyte:
            logger.debug("Reading chunk seed at {}".format(self.file.tell()))
            shape = (self.seedbyte, self.savedphoton)
            buffer = np.fromfile(self.file, dtype='B', count=np.prod(shape))
            seed = buffer.reshape(shape, order='F')
            seed = seed.transpose((0, 2, 1))
        else:
            seed = None

        return seed


    @staticmethod
    def read(file, offset=0, dtype='<f'):
        """Creates a new MCHFileChunk object and reads all data from the
        currently selected chunk in the mch file

        Parameters
        ----------
        file : str, file object
            Either the path to the tdms file to read as a string or
            pathlib.Path, or an already opened file.
        offset : uint64, optional
            Offset to the current position in file. Default is 0.
        dtype: data-type, optional
            The type of the statistical data stored in the file. Default is
            float with little endianness.
        """
        chunk = MCHFileChunk(file, offset, dtype)
        if not chunk.empty():
            chunk.loadData()
            chunk.loadSeed()
            file.seek(chunk.next_position)
        return chunk


    @staticmethod
    def readMetadata(file, offset=0, dtype='<f'):
        """Creates a new MCHFileChunk object and reads all meta data from the
        currently selected chunk in the mch file

        Parameters
        ----------
        file : str, file object
            Either the path to the tdms file to read as a string or
            pathlib.Path, or an already opened file.
        offset : uint64, optional
            Offset to the current position in file. Default is 0.
        dtype: data-type, optional
            The type of the statistical data stored in the file. Default is
            float with little endianness.
        """
        chunk = MCHFileChunk(file, offset, dtype)
        if not chunk.empty():
            file.seek(chunk.next_position)
        return chunk