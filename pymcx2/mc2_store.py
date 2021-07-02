# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Tue Mar 16 11:35:26 2021

@author: kpapke
"""
import os.path
import numpy as np
import pandas as pd

from .log import logmanager

logger = logmanager.getLogger(__name__)


__all__ = ["MC2Store", "load_mc2"]


class MC2Store(object):
    """ Reads and stores data from a mc2 file.

    There are two main ways to create a new MC2Store object.
    MC2Store.read will read all data into memory::

        store = MC2Store(mc2_file_path)
        store = MC2Store.read(mc2_file_path)

    or you can use MC2Store.open to read file metadata but not immediately read
    all data, for cases where a file is too large to easily fit in memory or
    you don't need to read data for all chunks:

        with MC2Store.open(mc2_file_path) as store:
            # Use store
            ...

    This class acts like a dictionary, where the keys are names of set flags in
    the mc2 files and the values are corresponding column(s).
    A MC2Store can be indexed by flag name to access the corresponding column(s)
    within the mc2 file, for example::

        with MC2Store.open(mc2_file_path) as store:
            col_data = store[flag_name]

    """

    def __init__(self, fname, mode='rb', dtype='<f'):
        """ Initialise a mc2 file object

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
        self.owner = False  # ownership of the underlying file
        self.mode = None  # read write mode
        self.dtype = dtype  # byte order

        # data and seed arrays
        self.data = None
        self.seed = None

        # keys to index data column wise
        self.keys = ["x", "y", "z", "time"]

        # mc2 file already open
        if hasattr(fname, "read") and hasattr(fname, "mode"):
            file = fname
            if file.mode == 'rb':
                self.file = file
                self.mode = file.mode
                self.owner = False  # no ownership of the underlying file
            else:
                raise ValueError(
                    "Unsupported mode to deal with file: {}.".format(file.mode))

        # open mc2 file
        elif os.path.isfile(str(fname)):
            if mode == 'rb':
                logger.debug("Open file object {}.".format(fname))
                self.file = open(fname, mode)
                self.mode = mode
                self.owner = True  # has ownership of the underlying file
            else:
                raise ValueError(
                    "Unsupported mode to open file: {}.".format(mode))

    def __enter__(self):
        logger.debug("MC2Store object __enter__().")
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("MC2Store object __exit__().")
        self.close()

    def __getitem__(self, index):
        """ Retrieve a mc2 group by name. """
        if self.data is not None and index < self.ntime:
            return self.data[..., index]
        else:
            return None

    def __len__(self):
        """ Returns the number of temporal points. """
        return self.ntime

    def _ensure_file_open(self):
        if self.file is None:
            raise RuntimeError(
                "Cannot read data after the underlying mc2 file is closed")

    def as_array(self):
        """ Return the entire data array. """
        return self.data

    def close(self):
        """ Close the underlying mc2 file if it was opened."""
        if self.file is not None and self.owner:
            logger.debug("Close file object.")
            self.file.close()

    def load_data(self, shape):
        """ Read all data from the mc2 file and store them internally.

        Parameters
        ----------
        shape : tuple or list of int,
            A tuple or list to specify the data dimension (nx, ny, nz, ntime).
        """
        self.data = self.read_data(shape)

    def read_data(self, shape, order='F'):
        """ Load the data of all chunks in the mc2 file.

        Parameters
        ----------
        shape : tuple or list of int,
            A tuple or list to specify the data dimension (nx, ny, nz, ntime).
        order : {'C', 'F', 'A'}, optional
            Read the elements of the file using this index order and place the
            elements into the reshaped array using this index order

        Returns
        -------
        data: np.ndarray
            Flux data stored in the mc2 file with the specified dimensions.
        """
        self._ensure_file_open()
        self.file.seek(0)

        if not isinstance(shape, (tuple, list)) or len(shape) != 4:
            raise ValueError("shape must be a tuple or list of four numbers")

        # read data
        logger.debug("Reading data at {}".format(self.file.tell()))
        buffer = np.fromfile(
            self.file, dtype=self.dtype, count=int(np.prod(shape)))
        if len(buffer) == int(np.prod(shape)):
            return buffer.reshape(shape, order=order)
        else:
            logger.debug("Cannot align data to the provided shape")
            return None

    @property
    def nx(self):
        """ int: The number of point in x direction."""
        return self.data.shape[0] if self.data is not None else None

    @property
    def ny(self):
        """ int: The number of point in y direction."""
        return self.data.shape[1] if self.data is not None else None

    @property
    def nz(self):
        """ int: The number of point in z direction."""
        return self.data.shape[2] if self.data is not None else None

    @property
    def ntime(self):
        """ int: The number of temporal points."""
        return self.data.shape[3] if self.data is not None else None

    @property
    def shape(self):
        """ int: The number of point in x direction."""
        return self.data.shape if self.data is not None else tuple([])

    @staticmethod
    def open(file_path, mode='rb', dtype='<f'):
        """ Creates a new mc2 File object and reads metadata, leaving the file
        open to allow reading data chunks

        Parameters
        ----------
        file_path : str
            The path to the mc2 file to read as a string or pathlib.Path.
        mode : str, optional
            The mode in which the file is opened. Default is "rb".
        dtype: data-type, optional
            The type of the statistical data stored in the file. Default is
            float with little endianness.
        """
        return MC2Store(file_path, mode=mode, dtype=dtype)

    @staticmethod
    def read(file_path, shape, dtype="<f"):
        """ Creates a new mc2 store object and reads all data from the file in.

        Parameters
        ----------
        file_path : str
            The path to the mc2 file to read as a string or pathlib.Path.
        shape : tuple or list of int,
            A tuple or list to specify the data dimension (nx, ny, nz, ntime).
        dtype: data-type, optional
            The type of the statistical data stored in the file. Default is
            float with little endianness.
        """
        store = None
        try:
            store = MC2Store.open(file_path, mode='rb', dtype=dtype)
            store.load_data(shape)  # load data from each chunk
        finally:
            if isinstance(store, MC2Store):
                store.close()  # close underlying file
        return store


def load_mc2(file_path, shape, dtype='<f'):
    """ Loads an mc2 file into a new MC2Store object.

    Parameters
    ----------
    file_path : str
        The path to the mc2 file to read as a string or pathlib.Path.
    shape : tuple or list of int,
            A tuple or list to specify the data dimension (nx, ny, nz, ntime).
    dtype: data-type, optional
        The type of the statistical data stored in the file. Default is
        float with little endianness.

    Returns
    -------
    store : :class:`MC2Store<pymcx2.MC2Store`
        A store object holding all date from the file.
    """
    with open(file_path, 'rb') as file:
        store = MC2Store(file, dtype=dtype)
        store.load_data(shape)  # load data from each chunk
    return store
