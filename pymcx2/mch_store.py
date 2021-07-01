# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Tue Mar 16 11:35:26 2021

@author: kpapke
"""
import os.path


import numpy as np
import pandas as pd

from .log import logmanager
from .mch_chunk import MCHFileChunk


logger = logmanager.getLogger(__name__)


__all__ = ["MCHStore", "load_mch"]


class MCHStore(object):
    """ Reads and stores data from a mch file.

    There are two main ways to create a new MCHStore object.
    MCHStore.read will read all data into memory::

        mch_file = MCHStore(mch_file_path)
        mch_file = MCHStore.read(mch_file_path)

    or you can use MCHStore.open to read file metadata but not immediately read
    all data, for cases where a file is too large to easily fit in memory or
    you don't need to read data for all chunks:

        with MCHStore.open(mch_file_path) as store:
            # Use store
            ...

    This class acts like a dictionary, where the keys are names of set flags in
    the mch files and the values are corresponding column(s).
    A MCHStore can be indexed by flag name to access the corresponding column(s)
    within the mch file, for example::

        with MCHStore.open(mch_file_path) as store:
            col_data = store[flag_name]

    """

    def __init__(self, fname, mode='rb', dtype='<f'):
        """ Initialise a mch file object

        Parameters
        ----------
        fname : file, str, or pathlib.Path
            The File, filepath, or generator to read.
        mode : str, optional
            The mode in which the file is opened. Default is "rb".
        dtype: data-type, optional
            The type of the statistical data stored in the file. Default is
            float with little endianness.
        """
        self.file = None  # underlying file pointer
        self.owner = False  # ownership for the underlying file
        self.mode = None  # read write mode

        # mch file chunks
        self.chunks = []

        # data and seed arrays
        self.dtype = np.dtype(dtype)  # type of statistical data
        self.data = None  # statistical data
        self.seed = None  # seeds

        # mch file already open
        if hasattr(fname, "read") and hasattr(fname, "mode"):
            file = fname
            if file.mode == 'rb':
                self.file = file
                self.mode = file.mode
                self.owner = False  # no ownership of the underlying file
            else:
                raise ValueError(
                    "Unsupported mode to deal with file: {}.".format(file.mode))

        # open mch file
        elif os.path.isfile(str(fname)):
            if mode == 'rb':
                logger.debug("Open file object {}.".format(fname))
                self.file = open(fname, mode)
                self.mode = mode
                self.owner = True  # has ownership of the underlying file
            else:
                raise ValueError(
                    "Unsupported mode to open file: {}.".format(mode))

        # load chunks (metadata only)
        self.load_chunks()

    def __enter__(self):
        logger.debug("MCHStore object __enter__().")
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("MCHStore object __exit__().")
        self.close()

    def __getitem__(self, key):
        """ Retrieve a TDMS group from the file by name. """
        if key in self.keys and self.index[key] is not None:
            index = self.index[key]  # index of first column
            ncol = len(self.cols[key])  # number of columns
            if ncol > 1:
                return self.data[:, index:index+ncol]
            elif ncol == 1:
                return self.data[:, index]

        else:
            raise KeyError("There is no key named '%s' in the mch file" % key)

    def __len__(self):
        """ Returns the number of saved photons. """
        return self.savedphoton

    def _ensure_file_open(self):
        if self.file is None:
            raise RuntimeError(
                "Cannot read data after the underlying mch file is closed")

    def as_array(self):
        return self.data

    def as_dataframe(self):
        """ Creates a dataframe from the data of the mch file. """
        columns = []
        for key in self.keys:
            columns.extend(self.cols[key])

        return pd.DataFrame(self.data, columns=columns)

    def clear(self):
        self.chunks.clear()

    def close(self):
        """ Close the underlying mch file if it was opened."""
        if self.file is not None and self.owner:
            logger.debug("Close file object.")
            self.file.close()

    def load_chunks(self):
        """ Load all chunks in the mch file. Read in only metadata."""
        self._ensure_file_open()

        self.file.seek(0)  # set the pointer to the beginning
        chunk = MCHFileChunk.read_metadata(self.file, 0, self.dtype)
        while not chunk.empty():
            self.chunks.append(chunk)
            chunk = MCHFileChunk.read_metadata(self.file, 0, self.dtype)
        self.file.seek(0)  # Again set the pointer to the beginning

    def load_data(self):
        """ Load the data of all chunks in the mch file."""
        self._ensure_file_open()

        if len(self.chunks):
            self.data = np.vstack([chunk.read_data() for chunk in self.chunks])
        else:
            self.data = None

    def load_seed(self):
        """ Load the seeds of all chunks in the mch file."""
        self._ensure_file_open()

        if len(self.chunks) and self.seedbyte:
            self.seed = np.vstack([chunk.read_seed() for chunk in self.chunks])
        else:
            self.seed = None

    @property
    def keys(self):
        """ list: List of keys to access statistical data of the array."""
        if len(self.chunks):
            return [key for key, flag in self.flags.items() if flag]
        else:
            return None

    @property
    def cols(self):
        """ dict: Column names for each key."""
        return self.chunks[0].cols if len(self.chunks) else None

    @property
    def index(self):
        """ dict: Data array column indices for each key."""
        return self.chunks[0].index if len(self.chunks) else None

    @property
    def detectedphoton(self):
        """ int: The number of sources."""
        return sum([chunk.detectedphoton for chunk in self.chunks]) if len(
            self.chunks) else None

    @property
    def flags(self):
        """ int: The number of detectors."""
        return self.chunks[0].flags if len(self.chunks) else None

    @property
    def lengthunit(self):
        """ int: # length unit to scale partial paths."""
        return self.chunks[0].lengthunit if len(self.chunks) else None

    @property
    def ndet(self):
        """int: The number of detectors."""
        return self.chunks[0].ndet if len(self.chunks) else None

    @property
    def nmat(self):
        """ int: The number of mediums."""
        return self.chunks[0].lengthunit if len(self.chunks) else None

    @property
    def normalizer(self):
        """ int: The scale factor."""
        return self.chunks[0].normalizer if len(self.chunks) else None

    @property
    def nsrc(self):
        """ int: The number of sources."""
        return self.chunks[0].nsrc if len(self.chunks) else None

    @property
    def respin(self):
        """ int: The respin number."""
        return self.chunks[0].respin if len(self.chunks) else None

    @property
    def savedphoton(self):
        """int: The number of sources."""
        return sum([chunk.savedphoton for chunk in self.chunks]) if len(
            self.chunks) else None

    @property
    def seedbyte(self):
        """ int: The number of seed bytes."""
        return self.chunks[0].seedbyte if len(self.chunks) else None

    @property
    def totalphoton(self):
        """ int: The number of sources."""
        return sum([chunk.totalphoton for chunk in self.chunks]) if len(
            self.chunks) else None

    @property
    def version(self):
        """ int: The mch file version."""
        return self.chunks[0].version if len(self.chunks) else None

    @staticmethod
    def open(file_path, mode='rb', dtype='<f'):
        """ Creates a new mch File object and reads metadata, leaving the file
        open to allow reading data chunks

        Parameters
        ----------
        file_path : str
            The path to the mch file to read as a string or pathlib.Path.
        mode : str, optional
            The mode in which the file is opened. Default is "rb".
        dtype: data-type, optional
            The type of the statistical data stored in the file. Default is
            float with little endianness.
        """
        return MCHStore(file_path, mode=mode, dtype=dtype)

    @staticmethod
    def read(file_path, dtype="<f"):
        """ Creates a new mch store object and reads all data from the file in.

        Parameters
        ----------
        file_path : str
            The path to the mch file to read as a string or pathlib.Path.
        dtype: data-type, optional
            The type of the statistical data stored in the file. Default is
            float with little endianness.
        """
        store = None
        try:
            store = MCHStore.open(file_path, mode='rb', dtype=dtype)
            store.load_data()  # load data from each chunk
            store.load_seed()  # load seed from each chunk
        finally:
            if isinstance(store, MCHStore):
                store.close()  # close underlying file
        return store


def load_mch(file_path, dtype='<f'):
    """ Loads an mch file into a new MCHStore object.

    Parameters
    ----------
    file_path : str
        The path to the mch file to read as a string or pathlib.Path.
    dtype: data-type, optional
        The type of the statistical data stored in the file. Default is
        float with little endianness.

    Returns
    -------
    store : :class:`MCHStore<pymcx2.MCHStore`
        A store object holding all date from the file.
    """
    with open(file_path, 'rb') as file:
        store = MCHStore(file, dtype=dtype)
        store.load_data()  # load data from each chunk
        store.load_seed()  # load seed from each chunk
    return store
