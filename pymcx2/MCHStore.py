# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:35:26 2021

@author: kpapke
"""
import os.path
import numpy as np

from .log import log_manager
from .MCHFileChunk import MCHFileChunk

__all__ = ['MCHStore']

logger = log_manager.get_logger(__name__)


class MCHStore(object):
    """ Reads and stores data from a mch file.

    There are two main ways to create a new MCHFile object.
    MCHFile.read will read all data into memory::

        mch_file = MCHFile(mch_file_path)
        mch_file = MCHFile.read(mch_file_path)

    or you can use MCHFile.open to read file metadata but not immediately read
    all data, for cases where a file is too large to easily fit in memory or
    you don't need to read data for all chunks:

        with MCHFile.open(mch_file_path) as mch_file:
            # Use mch_file
            ...

    This class acts like a dictionary, where the keys are names of set flags in
    the mch files and the values are corresponding column(s).
    A MCHFile can be indexed by flag name to access the corresponding column(s)
    within the mch file, for example::

        with MCHFile.open(mch_file_path) as mch_file:
            col_data = mch_file[flag_name]

    """

    def __init__(self, filePath, mode="rb", endianness="<"):
        """Initialise a mch file object

        Parameters
        ----------
        filePath : str
            The path to the mch file to read as a string or pathlib.Path
        mode : str, optional
            The mode in which the file is opened. Default is "rb".
        endianness: str, optional
            The byte order. Little-endian corresponds to '<'. Big-endian
            corresponds to '>'. Default is little-endian.
        """
        self.filePath = None
        self.file = None

        # opening mode
        if mode in ("rb"):
            self.mode = mode
        else:
            raise ValueError("Unsupported mode to open file: {}.".format(mode))

        # set byte order
        if endianness in ("<", "little", "ieee-le"):
            self.endianness = "<"
        elif endianness in (">", "big", "ieee-be"):
            self.endianness = ">"
        else:
            raise ValueError("Unknown byte order {}".format(endianness))

        self.chunks = []
        self.keys = []

        # metadata
        # self.version = None  # version of the mch file
        #
        # self.ncol = None  # number of columns in the raw data array
        # self.nmat = None  # number of mediums
        # self.ndet = None  # number of detectors
        # self.nsrc = None  # number of sources
        #
        # self.totalphoton = None  # total number of launched photons
        # self.detectedphoton = None  # number of detected photons
        # self.savedphoton = None  # number of saved photons
        #
        # self.flags = None  # to fill up bytes
        #
        # self.lengthunit = None  # length unit to scale partial paths
        # self.seedbyte = None  #
        # self.normalizer = None
        # self.respin = None

        # data arrays
        self.data = None
        self.seed = None

        # open file and load metadata of chunks
        if os.path.isfile(filePath):
            self.filePath = filePath
            self.file = open(self.filePath, self.mode)
            self.loadChunks()


    def __enter__(self):
        logger.debug("MCHFile object __enter__().")
        return self


    def __exit__(self, exception_type, exception_value, traceback):
        logger.debug("MCHFile object __exit__().")
        self.close()


    def __getitem__(self, key):
        """ Retrieve a TDMS group from the file by name
        """
        if key in self.keys:
            pass
        else:
            raise KeyError("There is no key named '%s' in the mch file" % key)


    def __len__(self):
        """ Returns the number of saved photons
        """
        return self.savedphoton


    def _ensureFileOpen(self):
        if self.file is None:
            raise RuntimeError(
                "Cannot read data after the underlying mch file is closed")


    def loadChunks(self):
        """Load all chunks in the mch file. Read in only metadata."""
        self._ensureFileOpen()

        self.file.seek(0)  # set the pointer to the beginning
        chunk = MCHFileChunk.readMetadata(self.file, 0, self.endianness)
        while not chunk.empty():
            self.chunks.append(chunk)
            chunk = MCHFileChunk.readMetadata(self.file, 0, self.endianness)
        self.file.seek(0)  # Again set the pointer to the beginning


    def loadData(self):
        """Load the data of all chunks in the mch file."""
        if len(self.chunks):
            self.data = np.vstack([chunk.readData() for chunk in self.chunks])
        else:
            self.data = None


    def loadSeed(self):
        """Load the seeds of all chunks in the mch file."""
        if len(self.chunks) and self.seedbyte:
            self.seed = np.vstack([chunk.readSeed() for chunk in self.chunks])
        else:
            self.seed = None


    def close(self):
        """ Close the underlying mch file if it was opened."""
        if self.file is not None:
            self.file.close()


    @property
    def detectedphoton(self):
        """int: The number of sources."""
        return sum([chunk.detectedphoton for chunk in self.chunks]) if len(
            self.chunks) else None

    @property
    def flags(self):
        """int: The number of detectors."""
        return self.chunks[0].flags if len(self.chunks) else None

    @property
    def lengthunit(self):
        """int: # length unit to scale partial paths."""
        return self.chunks[0].lengthunit if len(self.chunks) else None

    @property
    def ndet(self):
        """int: The number of detectors."""
        return self.chunks[0].ndet if len(self.chunks) else None

    @property
    def nmat(self):
        """int: The number of mediums."""
        return self.chunks[0].lengthunit if len(self.chunks) else None

    @property
    def normalizer(self):
        """int: The scale factor."""
        return self.chunks[0].normalizer if len(self.chunks) else None

    @property
    def nsrc(self):
        """int: The number of sources."""
        return self.chunks[0].nsrc if len(self.chunks) else None

    @property
    def respin(self):
        """int: The respin number."""
        return self.chunks[0].respin if len(self.chunks) else None

    @property
    def savedphoton(self):
        """int: The number of sources."""
        return sum([chunk.savedphoton for chunk in self.chunks]) if len(
            self.chunks) else None

    @property
    def seedbyte(self):
        """int: The number of seed bytes."""
        return self.chunks[0].seedbyte if len(self.chunks) else None

    @property
    def totalphoton(self):
        """int: The number of sources."""
        return sum([chunk.totalphoton for chunk in self.chunks]) if len(
            self.chunks) else None

    @property
    def version(self):
        """int: The mch file version."""
        return self.chunks[0].version if len(self.chunks) else None


    @staticmethod
    def open(filePath, mode='rb', endianness='<'):
        """ Creates a new mch File object and reads metadata, leaving the file
        open to allow reading data chunks

        Parameters
        ----------
        filePath : str
            The path to the mch file to read as a string or pathlib.Path.
        mode : str, optional
            The mode in which the file is opened. Default is "rb".
        endianness: str, optional
            The byte order. Little-endian corresponds to '<'. Big-endian
            corresponds to '>'. Default is little-endian.
        """
        return MCHStore(filePath, mode='rb', endianness=endianness)


    @staticmethod
    def read(filePath, endianness='<'):
        """ Creates a new TdmsFile object and reads all data in the file

        Parameters
        ----------
        filePath : str
            The path to the mch file to read as a string or pathlib.Path.
        endianness: str, optional
            The byte order. Little-endian corresponds to '<'. Big-endian
            corresponds to '>'. Default is little-endian.
        """
        store = MCHStore.open(filePath, mode='rb', endianness=endianness)
        store.loadData()  # load data from each chunk
        store.loadSeed()  # load seed from each chunk
        store.close()  # close underlying file
        return store




    # def as_dataframe(self, time_index=False, absolute_time=False, scaled_data=True):
    #     """
    #     Converts the TDMS file to a DataFrame. DataFrame columns are named using the TDMS object paths.
    #
    #     :param time_index: Whether to include a time index for the dataframe.
    #     :param absolute_time: If time_index is true, whether the time index
    #         values are absolute times or relative to the start time.
    #     :param scaled_data: By default the scaled data will be used.
    #         Set to False to use raw unscaled data.
    #         For DAQmx data, there will be one column per DAQmx raw scaler and column names will include the scale id.
    #     :return: The full TDMS file data.
    #     :rtype: pandas.DataFrame
    #     """
    #
    #     return pandas_export.from_tdms_file(self, time_index, absolute_time, scaled_data)
    #
    # def as_hdf(self, filepath, mode='w', group='/'):
    #     """
    #     Converts the TDMS file into an HDF5 file
    #
    #     :param filepath: The path of the HDF5 file you want to write to.
    #     :param mode: The write mode of the HDF5 file. This can be 'w' or 'a'
    #     :param group: A group in the HDF5 file that will contain the TDMS data.
    #     """
    #     return hdf_export.from_tdms_file(self, filepath, mode, group)

    # def data_chunks(self):
    #     """ A generator that streams chunks of data from disk.
    #     This method may only be used when the TDMS file was opened without reading all data immediately.
    #
    #     :rtype: Generator that yields :class:`DataChunk` objects
    #     """
    #     channel_offsets = defaultdict(int)
    #     for chunk in self._reader.read_raw_data():
    #         _convert_data_chunk(chunk, self._raw_timestamps)
    #         yield DataChunk(self, chunk, channel_offsets)
    #         for path, data in chunk.channel_data.items():
    #             channel_offsets[path] += len(data)




    #
    # def _read_file(self, tdms_reader, read_metadata_only, keep_open):
    #     tdms_reader.read_metadata(require_segment_indexes=keep_open)
    #
    #     # Use object metadata to build group and channel objects
    #     group_properties = OrderedDict()
    #     group_channels = OrderedDict()
    #     object_properties = {
    #         path_string: self._convert_properties(obj.properties)
    #         for path_string, obj in tdms_reader.object_metadata.items()}
    #     try:
    #         self._properties = object_properties['/']
    #     except KeyError:
    #         pass
    #
    #     for (path_string, obj) in tdms_reader.object_metadata.items():
    #         properties = object_properties[path_string]
    #         path = ObjectPath.from_string(path_string)
    #         if path.is_root:
    #             pass
    #         elif path.is_group:
    #             group_properties[path.group] = properties
    #         else:
    #             # Object is a channel
    #             try:
    #                 channel_group_properties = object_properties[path.group_path()]
    #             except KeyError:
    #                 channel_group_properties = OrderedDict()
    #             channel = TdmsChannel(
    #                 path, obj.data_type, obj.scaler_data_types, obj.num_values,
    #                 properties, channel_group_properties, self._properties,
    #                 tdms_reader, self._raw_timestamps, self._memmap_dir)
    #             if path.group in group_channels:
    #                 group_channels[path.group].append(channel)
    #             else:
    #                 group_channels[path.group] = [channel]
    #
    #     # Create group objects containing channels and properties
    #     for group_name, properties in group_properties.items():
    #         try:
    #             channels = group_channels[group_name]
    #         except KeyError:
    #             channels = []
    #         group_path = ObjectPath(group_name)
    #         self._groups[group_name] = TdmsGroup(group_path, properties, channels)
    #     for group_name, channels in group_channels.items():
    #         if group_name not in self._groups:
    #             # Group with channels but without any corresponding object metadata in the file:
    #             group_path = ObjectPath(group_name)
    #             self._groups[group_name] = TdmsGroup(group_path, {}, channels)
    #
    #     if not read_metadata_only:
    #         self._read_data(tdms_reader)
    #
    # def _read_data(self, tdms_reader):
    #     with Timer(log, "Allocate space"):
    #         # Allocate space for data
    #         for group in self.groups():
    #             for channel in group.channels():
    #                 self._channel_data[channel.path] = get_data_receiver(
    #                     channel, len(channel), self._raw_timestamps, self._memmap_dir)
    #
    #     with Timer(log, "Read data"):
    #         # Now actually read all the data
    #         for chunk in tdms_reader.read_raw_data():
    #             for (path, data) in chunk.channel_data.items():
    #                 channel_data = self._channel_data[path]
    #                 if data.data is not None:
    #                     channel_data.append_data(data.data)
    #                 elif data.scaler_data is not None:
    #                     for scaler_id, scaler_data in data.scaler_data.items():
    #                         channel_data.append_scaler_data(scaler_id, scaler_data)
    #
    #         for group in self.groups():
    #             for channel in group.channels():
    #                 channel_data = self._channel_data[channel.path]
    #                 if channel_data is not None:
    #                     channel._set_raw_data(channel_data)
    #
    #     self.data_read = True
    #
    # def _convert_properties(self, properties):
    #     def convert_prop(val):
    #         if isinstance(val, TdmsTimestamp) and not self._raw_timestamps:
    #             # Convert timestamps to numpy datetime64 if raw timestamps are not requested
    #             return val.as_datetime64()
    #         return val
    #     return OrderedDict((k, convert_prop(v)) for (k, v) in properties.items())