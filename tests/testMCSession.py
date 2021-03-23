# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Tue Mar 23 15:37:11 2021

@author: kai
"""
import sys
import os.path

import json
import numpy as np
import matplotlib.pyplot as plt

import logging

# add path to make mcx visible
# mcx = os.path.join(os.path.dirname(__file__), "..", "..", "mcx")
mcx = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "mcx"))
sys.path.append(os.path.abspath(mcx))

import utils
from pymcx2 import findMCX
from pymcx2 import MCSession
from pymcx2.log import logmanager

logger = logmanager.getLogger(__name__)


def main():
    data_path = os.path.join(os.getcwd(), "..", "model")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # load configuration ......................................................

    session = MCSession('benchmark_1x', workdir=data_path)

    vol = np.ones((200, 200, 11))
    vol[..., 0] = 0

    session.setDomain(vol, originType=1, lengthUnit=0.02)
    session.addMaterial(mua=1, mus=9, g=0.75, n=1)
    session.addMaterial(mua=0, mus=0, g=1, n=1)

    session.setBoundary(specular=True, missmatch=True, n0=1)
    session.setSource(
        nphoton=5e5, pos=[100, 100, 0], dir=[0, 0, 1], seed=29012392)
    session.addDetector(pos=[50, 50, 0], radius=50)

    session.setOutput(type="E", normalize=True, mask="DSPMXVW")
    session.dumpJSON()
    session.dumpVolume()

    session.run()



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
