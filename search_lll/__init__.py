# --- Unbuffered progress output -------------------------------------------
# When stdout is a pipe or a file (e.g. `sage search7_genus2.sage > run.txt`
# or `| tee`), Python block-buffers it, so nothing appears until ~8KB has
# accumulated -- which for the long residue-graph phase can be many minutes.
# Force line buffering once, here, at package import, so every print() in
# the package (and in forked ProcessPoolExecutor workers, which inherit this
# stream) shows up as it happens, instead of hand-adding flush=True to every
# call. Guarded because reconfigure() only exists on real TextIOWrapper
# streams (not, e.g., an IPython/Jupyter capture object).
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError, OSError):
        pass
del _sys, _stream

from .search_config import *
from .modularthread import *
from .rational_arithmetic import *
from .ll_utilities import *
from .search_analysis import *
from .search_main import *
from .diagnostics_univariate import *
from . import mumford, jacobian_basis
from .homology import *
from .selmer_genus2 import *
from .fiber_augment import *
from .fiber_augment_hdf5 import *
from .lp_incidence_dlp import *

"""
__init__.py: Exposes key functions from the submodules.
"""
# Expose core configuration and exceptions

# Expose main execution functions
# Expose main utilities (as needed by the parent scripts)
