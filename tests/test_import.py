# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Thu Jun 17 11:05:01 2021

pymcx2 searches during the import for the mxc binary through in all python
paths and provides a warning if not available

@author: kpapke
"""
from pymcx2 import findMCX

mcx_path = findMCX()




