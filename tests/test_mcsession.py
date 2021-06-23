# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Tue Mar 23 15:37:11 2021

@author: kpapke
"""
import sys

import os.path
import numpy as np

# windows system
if sys.platform == "win32":
    mcx = os.path.join("d:", os.path.sep, "projects", "mcx")

# linux system
else:
    mcx = os.path.join(os.path.expanduser("~"), "projects", "mcx")

sys.path.append(os.path.abspath(mcx))

from pymcx2 import MCSession, find_mcx


def test_find_mcx():
    """ Test whether mcx is found in sys.path. """
    assert find_mcx() is not None


def test_run_mcx(tmpdir):
    """ Test run for mcx. """
    data_path = tmpdir.mkdir("models")

    """ Test whether pymcx2 creates a correct json config file. """
    vol = np.ones((200, 200, 11))
    vol[..., 0] = 0  # pad a layer of zeros to get diffuse reflectance

    session = MCSession("test_mcx", workdir=data_path, seed=29012392)

    session.set_domain(vol, origin_type=1, scale=0.02)

    # background material with tag 0 is predefined with mua=0, mus=0, g=1, n=1
    session.add_material(mua=1, mus=9, g=0.75, n=1)  # receives tag 1
    session.add_material(mua=0, mus=0, g=1, n=1)  # receives tag 2

    session.set_boundary(specular=True, mismatch=True, n0=1)
    session.set_source(nphoton=500000, pos=[100, 100, 0], dir=[0, 0, 1])
    session.set_source_type(type='pencil')
    session.add_detector(pos=[50, 50, 0], radius=50)  # optional detector

    session.set_output(type="E", normalize=True, mask="DSPMXVW")

    session.run(thread='auto')  # , blocksize=64)

    # simulation statistics
    assert session.stat["normalizer"] == 2e-6
    assert session.stat["energytot"] == 500000
    assert session.stat["energyabs"] == 120765.25

    # fluence
    data = session.fluence[..., 0]  # last index refers to time step

    n0 = 1.0  # refractive index of surrounding
    n1 = 1.0  # refractive index of medium where photons enter
    spec_refl = ((n0 - n1) / (n0 + n1)) ** 2  # specular reflectance
    diff_refl = np.abs(np.sum(data[..., 0])) * (1 - spec_refl)
    absorbed = np.sum(data[:, :, 1:]) * (1 - spec_refl)

    assert np.round(absorbed, 4) == np.round(0.241532, 4)  # diffuse reflect
    assert np.round(diff_refl, 4) == np.round(0.097653, 4)  # absorbed fraction

    # detected photons
    dp = session.detected_photons

    assert len(dp) == 5797  # number of detected photons