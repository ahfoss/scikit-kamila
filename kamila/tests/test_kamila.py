"""Minimal stub tests for KamilaClustering."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

from kamila import KamilaClustering


def test_kamila_initialization():
    """Verify KamilaClustering can be instantiated with default and custom params."""
    kam = KamilaClustering()
    assert kam.n_clusters == 2
    assert kam.n_init == 10
    assert kam.max_iter == 25
    assert kam.cat_bandwidth == 0.025
    assert kam.random_state is None


def test_kamila_methods():
    """Verify stub methods fit, predict, and fit_predict."""
    X = [[1.0, 2.0], [3.0, 4.0]]
    kam = KamilaClustering()
    fitted = kam.fit(X)
    assert fitted is kam
    pred = kam.predict(X)
    assert pred is None
    fit_pred = kam.fit_predict(X)
    assert fit_pred is None


def test_kamila_cpp_extension():
    """Verify nanobind C++ extension is compiled and importable."""
    from kamila import _kamila_cpp

    assert _kamila_cpp.get_cpp_version() >= 1


def test_kamila_version():
    """Verify package exposes __version__ string."""
    import kamila

    assert hasattr(kamila, "__version__")
    assert isinstance(kamila.__version__, str)

