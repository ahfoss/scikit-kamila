"""KAMILA: KAy-means for MIxed LArge datasets clustering."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

from sklearn.base import BaseEstimator, ClusterMixin


class KamilaClustering(BaseEstimator, ClusterMixin):
    """KAMILA clustering of mixed-type data (stub).

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
    n_init : int, default=10
        Number of initializations.
    max_iter : int, default=25
        Maximum iterations per initialization.
    cat_bandwidth : float, default=0.025
        Categorical smoothing parameter.
    random_state : int, optional, default=None
        Random state seed.
    """

    def __init__(
        self,
        n_clusters=2,
        n_init=10,
        max_iter=25,
        cat_bandwidth=0.025,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.cat_bandwidth = cat_bandwidth
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model to X (stub).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : object
        """
        return self

    def predict(self, X):
        """Predict cluster labels for X (stub).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return None

    def fit_predict(self, X, y=None):
        """Fit and return cluster labels (stub)."""
        return self.fit(X, y).predict(X)
