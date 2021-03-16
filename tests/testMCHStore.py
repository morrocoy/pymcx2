# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:32:15 2021

@author: kai
"""
import sys
import os.path
import numpy as np
from struct import unpack
from scipy.interpolate import interp1d
import json

import logging

# import pymcx as mcx
from utils import loadmch, loadmc2
from pymcx2 import MCHStore

from pymcx2.log import log_manager

logger = log_manager.get_logger(__name__)


def main():
    data_path = os.path.join(os.getcwd(), "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    session = "benchmark_1"
    filePath = os.path.join(data_path, session + ".mch")

    # mch1 = mcx.loadmch(filePath, datadict=False)
    # mchdata = loadmch(filePath)

    mchstore = MCHStore.read(filePath)

    # print(mchstore.ncol)
    print("Number of medium: {}".format(mchstore.nmat))
    print("Number of detectors: {}".format(mchstore.ndet))
    print("Number of sources: {}".format(mchstore.nsrc))

    print("Total photons: {}".format(mchstore.totalphoton))
    print("Detected photons: {}".format(mchstore.detectedphoton))
    print("Saved photons: {}".format(mchstore.savedphoton))

    print("Flags: {}".format(mchstore.flags))

    print("Length unit: {}".format(mchstore.lengthunit))
    print("Seed bytes: {}".format(mchstore.seedbyte))
    print("Normalizer: {}".format(mchstore.normalizer))
    print("Respin: {}".format(mchstore.respin))

    # with MCHStore(filePath) as store:
    #     store.loadData()
    #     store.loadSeed()





if __name__ == '__main__':
    log_manager.set_level(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))

    # fmt = "%(asctime)s %(filename)35s: %(lineno)-4d: %(funcName)20s(): " \
    #       "%(levelname)-7s: %(message)s"
    # logging.basicConfig(level='DEBUG', format=fmt)

    # requests_logger = logging.getLogger('hsi')
    # requests_logger = logging.getLogger(__file__)
    # requests_logger.setLevel(logging.DEBUG)
    #
    # handler = logging.StreamHandler()
    # formatter = logging.Formatter(
    #         "%(asctime)s %(filename)35s: %(lineno)-4d: %(funcName)20s(): " \
    #           "%(levelname)-7s: %(message)s")
    # handler.setFormatter(formatter)
    # handler.setLevel(logging.DEBUG)
    # logger.addHandler(handler)
    # requests_logger.addHandler(handler)

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
#
#
# # load simulation results
# nbstep = round(
#     (cfg["Forward"]["T1"] - cfg["Forward"]["T0"]) / cfg["Forward"]["Dt"])
#
# # mesh shape (nx, ny, nz, nt)
# if "Dim" in cfg["Domain"] and cfg["Domain"]["Dim"] != []:
#     shape = cfg["Domain"]["Dim"] + [nbstep]
#
# elif "Shapes" in cfg:
#     for find in cfg["Shapes"]:
#         if "Grid" in find:
#             shape = find["Grid"]["Size"] + [nbstep]
#
# filePath = os.path.join(dirPaths['model'], "%s.mch" % (sid))
# # if os.path.isfile(filePath):
# #     mch = mcx.loadmch(filePath, datadict=True)

# session = "benchmark_1"
# filePath = os.path.join(dirPaths['data'], session + ".mch")
#
# # mch1 = mcx.loadmch(filePath, datadict=False)
# # mchdata = loadmch(filePath)
#
# with MCHStore(filePath) as store:
#     print("Hallo")

# d1 = mch1[0]

# d1 = mch1[0]
# d2 = mch2[0]
# np.array_equal(d1, d2)
#
# a = d2[:, 0]
# a = d2[0, :]
# data3 = mc3[..., 0]
#

# filePath = os.path.join(dirPaths['model'], "%s.mc2" % (sid))
# if os.path.isfile(filePath):
#     mc2 = loadmc2(filePath, shape)



# mc2 = loadmc2(filePath, shape, dtype=np.float32)
# mc3 = mcx.loadmc2(filePath, shape)
# data2 = mc2[..., 0]
# data3 = mc3[..., 0]
#
# np.array_equal(mc2, mc3)
#
# print("%.20f" % data2[0,0,0])
# print("%.20f" % data3[0,0,0])
# %timeit loadmc2(filePath, shape)
# %timeit mcx.loadmc2(filePath, shape)


# flags = 10
# flags = [1,2,3]
# flags[::-1]
# bflags =
# while val:
#     flags, bflag = divmod(flags, 2)
#     print(m)
#
# np.binary_repr(5, 8)