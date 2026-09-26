"""
===============================
Basic KAMILA Clustering Example
===============================

A complete example demonstrating how to instantiate, fit, and visualize
``KamilaClustering`` on mixed continuous and categorical data.
"""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from kamila import KamilaClustering

# 1. Generate synthetic mixed data (2 continuous features, 1 categorical feature)
rng = np.random.RandomState(42)
n_samples_per_cluster = 50

# Cluster 1
X_con_1 = rng.normal(loc=[1.0, 1.0], scale=0.6, size=(n_samples_per_cluster, 2))
X_cat_1 = rng.choice(["A", "B"], size=(n_samples_per_cluster, 1), p=[0.8, 0.2])

# Cluster 2
X_con_2 = rng.normal(loc=[5.0, 5.0], scale=0.6, size=(n_samples_per_cluster, 2))
X_cat_2 = rng.choice(["A", "B"], size=(n_samples_per_cluster, 1), p=[0.1, 0.9])

X_con = np.vstack([X_con_1, X_con_2])
X_cat = np.vstack([X_cat_1, X_cat_2])
X = np.hstack([X_con, X_cat])

# 2. Fit KAMILA clustering estimator
kamila = KamilaClustering(
    n_clusters=2,
    categorical_features=[2],
    n_init=5,
    random_state=42,
)
labels = kamila.fit_predict(X)

print("Fitted KAMILA Clustering:")
print(f"Number of iterations: {kamila.n_iter_}")
print(f"Continuous cluster centers:\n{kamila.cluster_centers_con_}")

# 3. Visualize continuous features colored by discovered cluster labels
fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(
    X[:, 0].astype(float),
    X[:, 1].astype(float),
    c=labels,
    cmap="viridis",
    alpha=0.8,
    edgecolors="k",
    label="Samples",
)
ax.scatter(
    kamila.cluster_centers_con_[:, 0],
    kamila.cluster_centers_con_[:, 1],
    c="red",
    s=120,
    marker="X",
    edgecolors="black",
    linewidths=1.5,
    label="Centers",
)
ax.set_title("KAMILA Clustering on Mixed Data")
ax.set_xlabel("Continuous Feature 1")
ax.set_ylabel("Continuous Feature 2")
ax.legend()
plt.tight_layout()
plt.show()
