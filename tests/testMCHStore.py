# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Wed Feb 24 18:32:15 2021

@author: kai
"""
import sys
import os.path

import numpy as np
import json
import logging

import utils
from pymcx2 import loadmch, loadmc2
from pymcx2 import MCHStore, MC2Store
from pymcx2.log import logmanager

logger = logmanager.getLogger(__name__)


def main():
    data_path = os.path.join(os.getcwd(), "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # load configuration ......................................................
    session = "benchmark_1"
    filePath = os.path.join(data_path, session + ".json")
    with open(filePath) as file:
        cfg = json.load(file)

    # get session id
    session = cfg['Session']['ID']

    # get mesh shape (nx, ny, nz, nt)
    ntime = round(
        (cfg["Forward"]["T1"] - cfg["Forward"]["T0"]) / cfg["Forward"]["Dt"])

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
    mchstore = loadmch(filePath)

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

    df = mchstore.asDataFrame()
    print(df)

    data2 = utils.loadmch(filePath)
    print(data2[0].shape)

    # load results of mc2 file ................................................
    filePath = os.path.join(data_path, session + ".mc2")

    # mc2store = MC2Store.read(filePath, meshShape)
    mc2store = loadmc2(filePath, meshShape)

    print("\nMetadata of mc2 file")
    print("-----------------------------------------------")
    print("Mesh shape: {}".format(mc2store.shape))
    print("Number of temporal points: {}".format(len(mc2store)))

    data = mc2store[0]




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
