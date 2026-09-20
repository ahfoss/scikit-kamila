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
