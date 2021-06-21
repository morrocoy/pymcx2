# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Wed Mar 24 08:36:24 2021

@author: kpapke
"""
import sys
import os.path

import numpy as np
import matplotlib.pyplot as plt

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

# define geometry .........................................................
vol = np.zeros((800, 800, 101))
vol[..., 0] = 0  # pad a layer of zeros to get diffuse reflectance
vol[..., 1:26] = 1  # first layer with material tag 1
vol[..., 26:51] = 2  # second layer with material tag 2
vol[..., 51:] = 3  # third layer with material tag 3


# configure and run simulation ............................................
session = MCSession('benchmark_3x', workdir=data_path, seed=29012392)

session.set_domain(vol, origin_type=1, length_unit=0.04)

# background material with tag 0 is predefined with mua=0, mus=0, g=1, n=1
session.add_material(mua=0.1, mus=10, g=0.9, n=1.37)  # receives tag 1
session.add_material(mua=0.1, mus=1, g=0.0, n=1.37)  # receives tag 2
session.add_material(mua=0.2, mus=1, g=0.7, n=1.37)  # receives tag 3
# overwrite background using 'set' command
session.set_material(tag=0, mua=0, mus=0, g=0, n=1)

session.set_boundary(specular=True, mismatch=True, n0=1)
session.set_source(nphoton=1e5, pos=[400, 400, 0], dir=[0, 0, 1])
session.set_source_type(type='pencil')
session.add_detector(pos=[50, 50, 0], radius=50)  # optional detector

session.set_output(type="E", normalize=True, mask="DSPMXVW")
session.dump_json()
session.dump_volume()

session.run(thread='auto', debug='P')

# post processing .........................................................
data = session.fluence[..., 0]  # last index refers to time step

n0 = 1.0  # refractive index of surrounding
n1 = 1.37  # refractive index of medium where photons enter
specrefl = ((n0 - n1) / (n0 + n1)) ** 2  # specular reflectance
diffrefl = np.abs(np.sum(data[..., 0])) * (1 - specrefl)  # diffuse reflect
absorbed = np.sum(data[:, :, 1:]) * (1 - specrefl)  # absorbed fraction
transmitted = 1 - diffrefl - absorbed - specrefl  # total transmittance

print("\nAbsorbed fraction: %f" % absorbed)
print("Diffuse reflectance: %f" % diffrefl)
print("Specular reflectance: %f" % specrefl)
print("Total transmittance: %f" % transmitted)

print("\nDetected photons:")
print(session.detected_photons)

# plot slice of fluence data ..............................................
dataSlice = data[:, :, 0]
dataSlice = np.log10(np.abs(dataSlice))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
pos = plt.imshow(dataSlice, vmin=-10, vmax=-2, cmap='jet')
# pos = ax.contourf(dataSlice, levels=np.arange(-10, -1, 1), cmap='jet')
fig.colorbar(pos, ax=ax)
ax.set_xlim([300, 500])
ax.set_ylim([300, 500])
plt.show()
plt.close()



