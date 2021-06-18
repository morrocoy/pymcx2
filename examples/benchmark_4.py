# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Wed Mar 24 08:36:24 2021

@author: kpapke

linux command GPU usage monitor:
watch -n 0.5 nvidia-smi
"""
import sys
import os.path
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hsi import HSTissueCompound

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

from pymcx2 import MCSession

data_path = os.path.join(os.getcwd(), "..", "model")

wavelen = np.linspace(500, 1000, 100, endpoint=False)
wavelen = np.linspace(500, 1000, 2, endpoint=False)

# tissue layers ..............................................................
p1 = {
    'blo': 0.0,  # blood
    'ohb': 0.0,  # oxygenated hemoglobin (O2HB)
    'hhb': 1.0,  # deoxygenation (HHB) - should be (1 - 'ohb')
    'methb': 0.,  # methemoglobin
    'cohb': 0.,  # carboxyhemoglobin
    'shb': 0.,  # sulfhemoglobin
    'wat': 0.0,  # water
    'fat': 0.0,  # fat
    'mel': 0.025,  # melanin
}

p2 = {
    'blo': 0.005,  # blood
    'ohb': 1.0,  # oxygenated hemoglobin (O2HB)
    'hhb': 0.0,  # deoxygenation (HHB) - should be (1 - 'ohb')
    'methb': 0.,  # methemoglobin
    'cohb': 0.,  # carboxyhemoglobin
    'shb': 0.,  # sulfhemoglobin
    'wat': 0.0,  # water
    'fat': 0.0,  # fat
    'mel': 0,  # melanin
}

layer1 = HSTissueCompound(portions=p1, skintype='epidermis', wavelen=wavelen)
layer2 = HSTissueCompound(portions=p2, skintype='dermis', wavelen=wavelen)

# define geometry ............................................................
plate_size = 100
vol = np.zeros((plate_size, plate_size, 202))
vol[..., 0] = 0  # pad a layer of zeros to get diffuse reflectance
vol[..., 1:2] = 1  # first layer with material tag 1
vol[..., 2:] = 2  # second layer with material tag 2

# configure and run simulation ............................................
session = MCSession('benchmark_4x', workdir=data_path, seed=29012392)
# session = MCSession('benchmark_4x', workdir=data_path, seed=-1)

session.set_domain(vol, origin_type=1, length_unit=0.1)

# dummy materials for each layer to be overwritten
# background material with tag 0 is predefined with mua=0, mus=0, g=1, n=1
session.add_material(mua=0, mus=0, g=1, n=1)  # receives tag 1
session.add_material(mua=0, mus=0, g=1, n=1)  # receives tag 2

# time-domain simulation parameters
session.set_forward(0, 1e-5, 1e-5)

# boundary conditions
session.set_boundary(specular=True, mismatch=True, n0=1)
session.set_source(
    nphoton=1e6, pos=[plate_size // 2, plate_size // 2, 0], dir=[0, 0, 1])
session.set_source_type(type='pencil')
# optional detector
# session.addDetector(
#     pos=[plate_size // 2, plate_size // 2, 0], radius=50)

# output settings
session.set_output(type="E", normalize=True, mask="DSPMXVW")

n = len(wavelen[:])
# n=1
data = np.zeros((n, 5))
# for i in [0,12]:#range(1):
for i in range(n):
    start = timer()
    print("Process wavelength: %d nm ... " % wavelen[i])#, end='')

    # overwrite material information of first layer
    session.set_material(
        tag=1,
        mua=layer1.absorption[i] / 10,  # 1/cm -> 1/mm
        mus=layer1.scattering[i] / 10,  # 1/cm -> 1/mm
        g=layer1.anisotropy[i],
        n=layer1.refraction[i]
    )

    # overwrite material information of second layer
    session.set_material(
        tag=2,
        mua=layer2.absorption[i] / 10,  # 1/cm -> 1/mm
        mus=layer2.scattering[i] / 10,  # 1/cm -> 1/mm
        g=layer2.anisotropy[i],
        n=layer2.refraction[i]
    )

    print("Layer1: mua=%f, mus=%f, g=%f, n=%f" %
          (layer1.absorption[i] / 10,  # 1/cm -> 1/mm
           layer1.scattering[i] / 10,  # 1/cm -> 1/mm
           layer1.anisotropy[i],
           layer1.refraction[i]))
    print("Layer2: mua=%f, mus=%f, g=%f, n=%f" %
          (layer2.absorption[i] / 10,  # 1/cm -> 1/mm
           layer2.scattering[i] / 10,  # 1/cm -> 1/mm
           layer2.anisotropy[i],
           layer2.refraction[i]))

    session.run(thread='auto', debug='P')
    # session.run(thread='auto', debug=0)

    # post processing
    if session.fluence is None:
        raise RuntimeError("No fluence data found.")

    fluence = session.fluence[..., 0]  # last index refers to time step

    # evaluate specular reflectance
    n0 = 1.0  # refractive index of surrounding
    n1 = layer1.refraction[i]  # refractive index of medium where photons enter
    specrefl = ((n0 - n1) / (n0 + n1)) ** 2

    # wavelength
    data[i, 0] = wavelen[i]
    # specular reflectance
    data[i, 1] = specrefl
    # diffuse reflectance
    data[i, 2] = np.abs(np.sum(fluence[..., 0])) * (1 - specrefl)
    # absorbed fraction
    data[i, 3] = np.sum(fluence[:, :, 1:]) * (1 - specrefl)
    # transmitted fraction
    data[i, 4] = 1 - np.sum(data[i, 1:4])

    print("Done. Elapsed time: %f sec" % (timer() - start))

    # df = session.detectedPhotons
    # df2 = df.loc[df['dir_exit_z'] < 0]
    # print(len(df2) / len(df))

# create and export a dataframe from results
df = pd.DataFrame(
    data, columns=["wavelen", "specular", "diffuse", "absorbed", "transmitted"])

print(df)
df.to_excel(os.path.join(data_path, "output.xls"))



