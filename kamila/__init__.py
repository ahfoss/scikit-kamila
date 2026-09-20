"""kamila: Clustering Mixed-Type Data in Python and Scikit-Learn."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

from ._kamila import KamilaClustering
from ._modha_spangler import ModhaSpanglerClustering
from .datasets import make_mixed_data

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0.dev0"

__all__ = [
    "KamilaClustering",
    "ModhaSpanglerClustering",
    "make_mixed_data",
    "__version__",
]
