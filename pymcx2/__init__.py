""" An interface module for mcx (Monte Carlo eXtreme).
"""

from __future__ import absolute_import


# Make version number available
from .version import __version_info__, __version__

# Export public objects
from .find_mcx import find_mcx
from .mch_chunk import MCHFileChunk
from .mch_store import MCHStore, load_mch
from .mc2_store import MC2Store, load_mc2
from .mc_session import MCSession, load_session
from .mc_store import MCStore

if find_mcx() is None:
    print("Warning: Could not find path to mcx binary.")