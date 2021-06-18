# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Wed Feb 24 18:32:15 2021

@author: kpapke
"""
import sys
import os.path

import json
import numpy as np
import matplotlib.pyplot as plt


import logging

import test_utils
from pymcx2 import load_mch, load_mc2
from pymcx2.log import logmanager

logger = logmanager.getLogger(__name__)


def main():
    data_path = os.path.join(os.getcwd(), "..", "model")

    # load configuration ......................................................
    session = "benchmark_1"
    filePath = os.path.join(data_path, session + ".json")
    with open(filePath) as file:
        cfg = json.load(file)

    # get session id
    session = cfg['Session']['ID']

    # get mesh shape (nx, ny, nz, nt)
    ntime = int(round(
        (cfg["Forward"]["T1"] - cfg["Forward"]["T0"]) / cfg["Forward"]["Dt"]))

    if "Dim" in cfg["Domain"] and cfg["Domain"]["Dim"] != []:
        meshShape = cfg["Domain"]["Dim"] + [ntime]
    elif "Shapes" in cfg:
        for find in cfg["Shapes"]:
            if "Grid" in find:
                meshShape = find["Grid"]["Size"] + [ntime]
    else:
        meshShape = tuple([])

    print("Session: {}".format(session))
    print("Mesh shape: {}".format(meshShape))

    # load results of mch file ................................................
    filePath = os.path.join(data_path, session + ".mch")

    # mchstore = MCHStore.read(filePath)
    mchstore = load_mch(filePath)

    print("\nMetadata of mch file")
    print("-----------------------------------------------")
    print("Number of medium: {}".format(mchstore.nmat))
    print("Number of detectors: {}".format(mchstore.ndet))
    print("Number of sources: {}".format(mchstore.nsrc))

    print("Total photons: {}".format(mchstore.totalphoton))
    print("Detected photons: {}".format(mchstore.detectedphoton))
    print("Saved photons: {}".format(mchstore.savedphoton))

    print("Flags: {}".format(mchstore.flags))
    print("Columns: {}".format(mchstore.cols))
    print("Keys: {}".format(mchstore.keys))
    print("Index: {}".format(mchstore.index))

    print("Length unit: {}".format(mchstore.lengthunit))
    print("Seed bytes: {}".format(mchstore.seedbyte))
    print("Normalizer: {}".format(mchstore.normalizer))
    print("Respin: {}".format(mchstore.respin))


    print(mchstore["detectid"])
    print(mchstore["ppathlen"])
    # print(mchstore['nscatter'])

    df = mchstore.as_dataframe()
    print(df)

    data2 = test_utils.loadmch(filePath)
    print(data2[0].shape)

    # load results of mc2 file ................................................
    filePath = os.path.join(data_path, session + ".mc2")

    # mc2store = MC2Store.read(filePath, meshShape)
    mc2store = load_mc2(filePath, meshShape)

    print("\nMetadata of mc2 file")
    print("-----------------------------------------------")
    print("Mesh shape: {}".format(mc2store.shape))
    print("Number of temporal points: {}".format(len(mc2store)))

    data = mc2store[0]

    # dslice = np.log10(data[200:300, 200:300, 1])
    dslice = np.log10(data[..., 1])
    # dslice = np.log10(data[400, ...])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    pos = plt.imshow(dslice, vmin=-10, vmax=-2, cmap='jet')
    fig.colorbar(pos, ax=ax)
    plt.show()
    plt.close()

    fig = plt.figure()
    # fig.set_size_inches(10, 8)
    ax = fig.add_subplot(1,1,1)
    pos = ax.contourf(dslice, levels=np.arange(-10, -1, 1), cmap='jet')
    fig.colorbar(pos, ax=ax)
    plt.show()
    plt.close()

    n1 = 1.0  # refractive index of surrounding
    n2 = 1.5  # refractive index of medium
    n2 = 1.37  # refractive index of medium
    specrefl = ((n1-n2)/(n1+n2))**2  # specular reflectance
    diffrefl = np.abs(np.sum(data[:, :, 0])) * (1-specrefl)   # diffuse reflectance
    absorbed = np.sum(data[:, :, 1:])  * (1-specrefl)  # absorbed fraction
    transmitted = 1 - diffrefl - absorbed - specrefl # total transmittance


    # energytot = 4800  # from command line output
    # nphoton = 5000  # from command line output
    # specrefl = (nphoton - energytot) / nphoton  # specular reflectance
    # diffrefl = np.abs(np.sum(data[:, :, 0])) * energytot / nphoton  # diffuse reflectance
    # absorbed = np.sum(data[:, :, 1:]) * energytot / nphoton # absorbed fraction
    # transmitted = 1 - diffrefl - absorbed - specrefl # total transmittance


    print("Absorbed fraction: %f" % absorbed)
    print("Diffuse reflectance: %f" % diffrefl)
    print("Specular reflectance: %f" % specrefl)
    print("Total transmittance: %f" % transmitted)



if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))

    main()


# fileName = "benchmark_3.json"

# filePath = os.path.join(dirPaths['model'], fileName)
# with open(filePath) as file:
#     cfg = json.load(file)
#
#     sid = cfg['Session']['ID']
#     print("Session: {}".format(cfg['Session']['ID']))
#
# # mcx.run(cfg, flag="", mcxbin=mcxbin)
#
#
# # ..\..\bin\mcx.exe -A -f benchmark3.json -b 1 -s benchmark_3 %*
#
# # cmd = mcxbin+' -A -f ' + filePath + " -b 0 %*"
# cmd = mcxbin + " -A -f " + filePath + " -b 1 -s " + sid + " %*"  # specify session id
# # cmd = mcxbin+' -f benchmark1.json'
#
# print(cmd)
# # os.system(cmd)
