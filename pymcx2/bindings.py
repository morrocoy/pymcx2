# -*- coding: utf-8 -*-
"""
Helper functions that smooth out the differences between python 2 and 3.
"""
import sys

if sys.version_info[0] == 3:
    from collections import OrderedDict as ordered_dict

else:
    # ordered_dict = dict
    from collections import OrderedDict as ordered_dict

