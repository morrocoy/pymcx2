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
# alternatively add the path via spyder's python path manager
# windows system
if sys.platform == "win32":
    mcx = os.path.join("d:", os.path.sep, "projects", "mcx")
    # mcx = os.path.join(os.path.dirname(__file__), "..", "..", "mcx")
    # mcx = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "mcx"))

# linux system
else:
    mcx = os.path.join(os.path.expanduser("~"), "projects", "mcx")

sys.path.append(os.path.abspath(mcx))

from pymcx2 import MCSession, load_session
from pymcx2.log import logmanager

logger = logmanager.getLogger(__name__)


def main():
    data_path = os.path.join(os.getcwd(), "..", "model")

    # define geometry .........................................................
    vol = np.ones((200, 200, 11))
    vol[..., 0] = 0

    # configure and run simulation ............................................
    session = MCSession("benchmark_1x", workdir=data_path, seed=29012392)
    session.clear_files()

    session.set_domain(vol, origin_type=1, length_unit=0.02)
    session.add_material(mua=1, mus=9, g=0.75, n=1)
    session.add_material(mua=0, mus=0, g=1, n=1)

    session.set_boundary(specular=True, mismatch=True, n0=1)
    session.set_source(nphoton=5e5, pos=[100, 100, 0], dir=[0, 0, 1])
    session.set_source_type(type="pencil")
    session.add_detector(pos=[50, 50, 0], radius=50)

    session.set_output(type="E", normalize=True, mask="DSPMXVW")
    session.dump_json()
    session.dump_volume()

    session.run(thread="auto", debug='P')

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

    # compare results with other session (python2 with python3 runs)
    session1 = load_session(os.path.join(data_path, "benchmark_1x2.json"))
    data1 = session1.fluence[..., 0]  # last index refers to time step

    df = session.detectedPhotons.sort_values(
        ["detectid", "nscatter_mat0", "nscatter_mat1", "ppathlen_mat0"])
    df.reset_index(drop=True, inplace=True)
    df1 = session1.detectedPhotons.sort_values(
        ["detectid", "nscatter_mat0", "nscatter_mat1", "ppathlen_mat0"])
    df1.reset_index(drop=True, inplace=True)

    df = df - df1

    print(np.max(np.abs(data1 - data)))
    print(np.max(np.abs(data1)))
    print(df.abs().max())



if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))

    main()