# -*- coding: utf-8 -*-
""" An interface module for mcx (Monte Carlo eXtreme).

Created on Thu Jun 17 19:59:04 2021

@author: kpapke

Helper functions that smooth out the differences between python 2 and 3.
"""
import sys

if sys.version_info < (3, 6):
    from collections import OrderedDict as ordered_dict

else:
    ordered_dict = dict


