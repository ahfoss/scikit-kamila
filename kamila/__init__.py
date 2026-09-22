"""kamila: Clustering Mixed-Type Data in Python and Scikit-Learn."""

from ._kamila import KamilaClustering

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.1.0.dev0"

__all__ = [
    "KamilaClustering",
    "__version__",
]
