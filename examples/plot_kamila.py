"""
===============================
Basic KAMILA Clustering Example
===============================

A minimal example demonstrating how to instantiate and use ``KamilaClustering``.
"""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
# License: BSD 3 clause

import numpy as np

from kamila import KamilaClustering

# Create a toy dataset with continuous features
rng = np.random.RandomState(42)
X = np.vstack([rng.randn(30, 2), rng.randn(30, 2) + 4.0])

# Initialize the KAMILA clustering estimator stub
kamila = KamilaClustering(n_clusters=2, random_state=42)
print("Instantiated:", kamila)
