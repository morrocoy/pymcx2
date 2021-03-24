# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Tue Mar 23 15:37:11 2021

@author: kpapke
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

from pymcx2 import MCSession
from pymcx2.log import logmanager

logger = logmanager.getLogger(__name__)


def main():
    data_path = os.path.join(os.getcwd(), "..", "model")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # define geometry .........................................................
    vol = np.ones((200, 200, 11))
    vol[..., 0] = 0

    # configure and run simulation ............................................
    session = MCSession('benchmark_1x', workdir=data_path, seed=29012392)

    session.setDomain(vol, originType=1, lengthUnit=0.02)
    session.addMaterial(mua=1, mus=9, g=0.75, n=1)
    session.addMaterial(mua=0, mus=0, g=1, n=1)

    session.setBoundary(specular=True, missmatch=True, n0=1)
    session.setSource(nphoton=5e5, pos=[100, 100, 0], dir=[0, 0, 1])
    session.setSourceType(type='pencil')
    session.addDetector(pos=[50, 50, 0], radius=50)

    session.setOutput(type="E", normalize=True, mask="DSPMXVW")
    session.dumpJSON()
    session.dumpVolume()

    session.run(thread='auto', debug='P')

    # post processing .........................................................
    data = session.fluence[..., 0]  # last index refers to time step

    n0 = 1.0  # refractive index of surrounding
    n1 = 1.0  # refractive index of medium where photons enter
    specrefl = ((n0 - n1) / (n0 + n1)) ** 2  # specular reflectance
    diffrefl = np.abs(np.sum(data[..., 0])) * (1 - specrefl)  # diffuse reflect
    absorbed = np.sum(data[:, :, 1:]) * (1 - specrefl)  # absorbed fraction
    transmitted = 1 - diffrefl - absorbed - specrefl  # total transmittance

    print("\nAbsorbed fraction: %f" % absorbed)
    print("Diffuse reflectance: %f" % diffrefl)
    print("Specular reflectance: %f" % specrefl)
    print("Total transmittance: %f" % transmitted)

    print("\nDetected photons:")
    print(session.detectedPhotons)

    # plot slice of fluence data ..............................................
    dataSlice = data[:, :, 0]
    dataSlice = np.log10(np.abs(dataSlice))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    pos = plt.imshow(dataSlice, vmin=-10, vmax=-2, cmap='jet')
    # pos = ax.contourf(dataSlice, levels=np.arange(-10, -1, 1), cmap='jet')
    fig.colorbar(pos, ax=ax)
    plt.show()
    plt.close()





if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))

    main()