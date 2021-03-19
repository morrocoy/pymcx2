# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:12:11 2020

@author: papkai
"""
import os.path
import re
from ast import literal_eval
import xlrd
from struct import unpack

import numpy as np
from scipy.interpolate import interp1d

def readExcelTbl(file, sheet, rows, cols, zerochars=[], nanchars=[]):

    workbook = xlrd.open_workbook(file)

    if isinstance(sheet, int):
        worksheet = workbook.sheet_by_index(sheet)
    else:
        worksheet = workbook.sheet_by_name(sheet)

    data = []
    for i in (range(len(rows))):
        line = []
        if worksheet.nrows <= rows[i]:
            break
        for j in range(len(cols)):
            readout = worksheet.cell_value(rows[i], cols[j])
            # check for number
            if isinstance(readout, (int, float)):
                if readout == int(readout):
                    line.append(int(readout))
                else:
                    line.append(readout)

            # check for characters which shall be associated with zero
            elif readout in zerochars:
                line.append(0)

            # check for characters which shall be associated with zero
            elif readout in nanchars:
                line.append(np.nan)

            # append readout as string
            else:
                line.append(readout)
        data.append(line)
    return data


def readMetadata(file):
    """Read metadata of a data chunk.

    Parameters
    ----------
    file : str
        The input file.

    Returns
    -------
    metadata : dict
        A dictionary that contains version, medianum, detnum, recordnum,
        totalphoton, detectedphoton, savedphoton, lengthunit, seed byte,
        normalize, respin.
    """
    buffer = file.read(4)  # a char is 1 Bytes
    if not buffer or buffer != b'MCXH':
        return None

    version = np.fromfile(file, dtype=np.uint32, count=1)[0]
    if version != 1:
        print("version higher than 1 is not supported")
        return None

    # read metadata
    nmat = np.fromfile(file, dtype=np.uint32, count=1)[0]
    ndet = np.fromfile(file, dtype=np.uint32, count=1)[0]
    ncol = np.fromfile(file, dtype=np.uint32, count=1)[0]
    totalphoton = np.fromfile(file, dtype=np.uint32, count=1)[0]
    detectedphoton = np.fromfile(file, dtype=np.uint32, count=1)[0]
    savedphoton = np.fromfile(file, dtype=np.uint32, count=1)[0]
    unitmm = np.fromfile(file, dtype=np.float32, count=1)[0]
    seedbyte = np.fromfile(file, dtype=np.uint32, count=1)[0]
    normalizer = np.fromfile(file, dtype=np.float32, count=1)[0]
    respin = np.fromfile(file, dtype=np.int32, count=1)[0]
    srcnum = np.fromfile(file, dtype=np.uint32, count=1)[0]
    savedetflag = np.fromfile(file, dtype=np.uint32, count=1)[0]
    junk = np.fromfile(file, dtype=np.int32, count=2)

    if respin > 1:
        totalphoton *= respin

    metadata = {
        'version': version,
        'nmat': nmat,  # number of media
        'ndet': ndet,  # number of detectors
        'ncol': ncol,  # number of columns of the raw data array
        'totalphoton': totalphoton,
        'detectedphoton': detectedphoton,
        'savedphoton': savedphoton,
        'lengthunit': unitmm,  # length scale
        'seedbyte': seedbyte,
        'normalizer': normalizer,
        'respin': respin,
        'srcnum': srcnum,
        'savedetflag': savedetflag
    }

    return metadata


def readDataChunk(file, format='f', endian='ieee-le'):
    """Load data chunk of a mch file.

    Parameters
    ----------
    file : str
        The input file.
    format : tuple or list of int,
        The datatype format
    endian : str
        A string to specify endian format.

    Returns
    -------
    data : np.ndarray
        The output detected photon data array with the following columns:
        [detid(1) nscat(M) ppath(M) mom(M) p(3) v(3) w0(1)]. Depending on the
        flags the data array contains the following information:

        Bit Flag Description
        === ==== ================================
        1   D    Detector ID (1)
        2   S    Scattering event counts (#media)
        4   P    Partial path-lengths (#media)
        8   M    Momentum transfer (#media)
        16  X    Exit position (3)
        32  V    Exit direction (3)
        64  W    Initial weight (1)


    seeds : np.ndarray
        If the mch file contains a seed section, this returns the seed data for
        each detected photon. Each row is a byte array, which can be used to
        initialize a seeded simulation. Note that the seed is RNG specific. You
        must use the identical RNG to utilize these seeds for a new simulation.
    """
    metadata = readMetadata(file)
    if metadata is None:
        return None, None

    # read data
    shape = (metadata['savedphoton'], metadata['ncol'])
    buffer = np.fromfile(file, dtype=format, count=np.prod(shape))
    buffer = buffer.reshape(shape)

    # unravel flags
    if endian == 'ieee-le':  # little-endian ordering
        detflag = np.binary_repr(metadata['savedetflag'] % 256, 8)[::-1]
    else:  # big-endian ordering
        detflag = np.binary_repr(metadata['savedetflag'] % 256, 8)

    # create output data array
    nmat = metadata['nmat']  # number of media
    offset = 0  # column offset for the raw data array
    data = np.zeros((metadata['savedphoton'], 3 * nmat + 8))
                    # dtype='i4, i4, f, f, f, f, f')

    # D - the id of the detector that captures the photon (1)
    if detflag[0] == '1':
        icol = 0
        data[:, icol] = buffer[:, offset]
        offset += 1

    # S - number of scattering events of detected photon (number of media)
    if detflag[1] == '1':
        icol = 1
        data[:, icol:icol + nmat] = buffer[:, offset:offset + nmat]
        offset += nmat

    # P - partial path length (in mm) for each medium type (number of media)
    if detflag[2] == '1':
        icol = 1 + nmat
        data[:, icol:icol + nmat] = buffer[:, offset:offset + nmat] #* metadata[
            # 'lengthunit']
        offset += nmat

    # M - momentum transfer for each medium type (number of media)
    if detflag[3] == '1':
        icol = 1 + 2 * nmat
        data[:, icol:icol + nmat] = buffer[:, offset:offset + nmat]
        offset += nmat

    # X - output exit position (3)
    if detflag[4] == '1':
        icol = 1 + 3 * nmat
        data[:, icol:icol + 3] = buffer[:, offset:offset + 3]
        offset += 3

    # V - output exit direction (3)
    if detflag[5] == '1':
        icol = 1 + 3 * nmat + 3
        data[:, icol:icol + 3] = buffer[:, offset:offset + 3]
        offset += 3

    # W - output initial weight (1)
    if detflag[6] == '1':
        icol = 1 + 3 * nmat + 6
        data[:, icol] = buffer[:, offset]
        offset += 1

    # photon seeds
    shape = (metadata['seedbyte'], metadata['savedphoton'])
    if metadata['seedbyte']:
        buffer = np.fromfile(file, dtype='B', count=np.prod(shape))
        seeds = buffer.reshape(shape, order='F')
        # seeds = seeds.transpose((0, 2, 1))
        seeds = seeds.transpose()
    else:
        seeds = None

    return data, seeds



def loadmch(filePath, format='f', endian='ieee-le'):
    """Load mc2 file.

    Parameters
    ----------
    filePath : str
        The path to the input file.
    format : tuple or list of int,
        The datatype format
    endian : str
        A string to specify endian format.
    datadict : bool, optional
        Enable or disable dictionary output

    Returns
    -------
    data : np.ndarray
        The output detected photon data array with the following columns:
        [detid(1) nscat(M) ppath(M) mom(M) p(3) v(3) w0(1)]. Depending on the
        flags the data array contains the following information:

        Bit Flag Description
        === ==== ================================
        1   D    Detector ID (1)
        2   S    Scattering event counts (#media)
        4   P    Partial path-lengths (#media)
        8   M    Momentum transfer (#media)
        16  X    Exit position (3)
        32  V    Exit direction (3)
        64  W    Initial weight (1)


    metadata : dict
        A dictionary that contains version, medianum, detnum, recordnum,
        totalphoton, detectedphoton, savedphoton, lengthunit, seed byte,
        normalize, respin.
    seeds : np.ndarray
        If the mch file contains a seed section, this returns the seed data for
        each detected photon. Each row is a byte array, which can be used to
        initialize a seeded simulation. Note that the seed is RNG specific. You
        must use the identical RNG to utilize these seeds for a new simulation.
    """
    if not os.path.isfile(filePath):
        return tuple([])

    # read metadata of the first chunk separately
    with open(filePath, "rb") as file:
        metadata = readMetadata(file)

    if metadata is None:
        return None, None, None

    data = []
    seeds = []
    with open(filePath, "rb") as file:
        dataChunk, seedsChunk = readDataChunk(file, format, endian)
        while dataChunk is not None:
            data.append(dataChunk)
            seeds.append(seedsChunk)
            dataChunk, seedsChunk = readDataChunk(file, format, endian)

    data = np.asarray(data).squeeze()
    seeds = np.asarray(seeds).squeeze()
    return data, metadata, seeds


def loadmc2(filePath, shape, dtype='<f'):
    """Load mc2 file.

    Parameters
    ----------
    filePath : str
        The path to the input file.
    shape : tuple or list of int,
        A tuple or list to specify the output data dimension (nx, ny, nz, nt).
    dtype :  data type of the values in the mcx solution data array.

    Returns
    -------
    data: np.ndarray
        The output MCX solution data array with the specified dimensions.
    """
    if not os.path.isfile(filePath):
        return tuple([])

    with open(filePath, 'rb') as file:
        buffer = file.read()
        data = np.ndarray(
            shape, dtype=dtype, buffer=buffer, offset=0, order='F')

    return data





