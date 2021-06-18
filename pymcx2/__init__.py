""" An interface module for mcx (Monte Carlo eXtreme).
"""

from __future__ import absolute_import


# Make version number available
from .version import __version_info__, __version__

# Export public objects
from .findmcx import find_mcx
from .mchchunk import MCHFileChunk
from .mchstore import MCHStore, load_mch
from .mc2store import MC2Store, load_mc2
from .mcsession import MCSession, load_session


if find_mcx() is None:
    print("Warning: Could not find path to mcx binary.")