"""Modha-Spangler clustering for mixed continuous and categorical data."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .utils._validation import extract_mixed_features


class ModhaSpanglerClustering(BaseEstimator, ClusterMixin):
    """Modha-Spangler clustering for mixed-type data.

    Estimates the optimal weighting for continuous vs categorical variables
    using a brute-force search strategy over normalized cluster distortions
    (Modha & Spangler, 2003; Foss & Markatou, 2018).

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
    search_density : int, default=10
        Number of distinct continuous weight values evaluated in the grid search.
    categorical_features : 'auto', list of column names/indices, boolean mask, or None, default=None
        Specification of categorical features.
    n_init : int, default=10
        Number of time the underlying k-means algorithm is run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    random_state : int, RandomState instance, default=None
        Determines random number generation for k-means initialization.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster assignments.
    best_weight_ : float
        Optimal continuous weight selected by the grid search.
    con_centers_ : ndarray of shape (n_clusters, n_con_features)
        Coordinates of continuous cluster centroids.
    cat_centers_ : ndarray of shape (n_clusters, n_dummy_features)
        Coordinates of categorical cluster centroids (in dummy space).
    best_objective_ : float
        Minimum objective function value achieved.
    weights_ : ndarray of shape (search_density,)
        Grid of weights evaluated.
    objective_values_ : ndarray of shape (search_density,)
        Objective function values for each evaluated weight.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.

    References
    ----------
    .. [1] Modha DS, Spangler WS. "Feature Weighting in k-Means Clustering."
           Machine Learning, 52(3), 217-237 (2003).
           doi: 10.1023/a:1024016609528
    .. [2] Foss A, Markatou M. "kamila: Clustering Mixed-Type Data in R and Hadoop."
           Journal of Statistical Software, 83(13), 1-45 (2018).
           doi: 10.18637/jss.v083.i13
    """

    def __init__(
        self,
        n_clusters: int = 2,
        search_density: int = 10,
        categorical_features: Optional[Union[str, Sequence[Union[int, str]], np.ndarray]] = None,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.n_clusters = n_clusters
        self.search_density = search_density
        self.categorical_features = categorical_features
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def _dummy_code(self, X_cat: np.ndarray, categories: List[np.ndarray]) -> np.ndarray:
        """One-hot encode categorical integer matrix."""
        n_samples = len(X_cat)
        dummies = []
        for q, cats in enumerate(categories):
            col = X_cat[:, q]
            n_lev = len(cats)
            d = np.zeros((n_samples, n_lev), dtype=np.float64)
            d[np.arange(n_samples), col] = 1.0
            dummies.append(d)
        return np.column_stack(dummies) if len(dummies) > 0 else np.empty((n_samples, 0), dtype=np.float64)

    def fit(self, X: Any, y: Any = None) -> "ModhaSpanglerClustering":
        """Compute Modha-Spangler clustering on mixed data.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : object
        """
        if self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}")
        if self.search_density < 1:
            raise ValueError(f"search_density must be >= 1, got {self.search_density}")

        X_con, X_cat, categories, con_idx, cat_idx, feat_names = extract_mixed_features(
            X, categorical_features=self.categorical_features, fit=True
        )

        if X_con is None or X_cat is None:
            raise ValueError(
                "ModhaSpanglerClustering requires both continuous and categorical variables."
            )

        n_samples = len(X_con)
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot exceed number of samples ({n_samples})"
            )

        self.con_indices_ = con_idx
        self.cat_indices_ = cat_idx
        self.categorical_levels_ = categories
        self.n_features_in_ = len(con_idx) + len(cat_idx)
        if feat_names is not None:
            self.feature_names_in_ = feat_names

        # Dummy encode categorical data
        X_cat_dum = self._dummy_code(X_cat, categories)

        # Check unique categorical combinations
        _, unique_idx = np.unique(X_cat, axis=0, return_index=True)
        if self.n_clusters >= len(unique_idx):
            raise ValueError(
                "n_clusters must be less than the number of unique categorical level combinations."
            )

        # Total continuous and categorical distortions (from 1-cluster centroid)
        con_mean = np.mean(X_con, axis=0, keepdims=True)
        total_con_dist = float(np.sum((X_con - con_mean) ** 2))

        cat_mean = np.mean(X_cat_dum, axis=0, keepdims=True)
        total_cat_dist = float(np.sum((X_cat_dum - cat_mean) ** 2))

        # Grid of continuous weights
        step = 1.0 / (1.0 + self.search_density)
        weights = np.linspace(step, 1.0 - step, self.search_density)

        obj_fun = np.full(len(weights), np.nan, dtype=np.float64)
        q_con = np.full(len(weights), np.nan, dtype=np.float64)
        q_cat = np.full(len(weights), np.nan, dtype=np.float64)

        best_obj = np.inf
        best_res = None
        best_w = None

        rng = check_random_state(self.random_state)

        for i, w in enumerate(weights):
            # Scale continuous by w, categorical by (1 - w)
            con_scaled = X_con * w
            cat_scaled = X_cat_dum * (1.0 - w)
            X_combined = np.hstack([con_scaled, cat_scaled])

            km = KMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
            )
            km.fit(X_combined)

            # Reconstitute cluster centers
            n_con = X_con.shape[1]
            con_centers = km.cluster_centers_[:, :n_con] / w
            cat_centers = km.cluster_centers_[:, n_con:] / (1.0 - w)

            # Calculate within-cluster continuous and categorical squared Euclidean distance
            within_con = 0.0
            within_cat = 0.0
            for cl in range(self.n_clusters):
                mask = km.labels_ == cl
                if np.any(mask):
                    within_con += float(np.sum((X_con[mask] - con_centers[cl]) ** 2))
                    within_cat += float(np.sum((X_cat_dum[mask] - cat_centers[cl]) ** 2))

            diff_con = total_con_dist - within_con
            diff_cat = total_cat_dist - within_cat

            q_c = 0.0 if within_con == 0.0 else (np.inf if diff_con <= 0.0 else within_con / diff_con)
            q_k = 0.0 if within_cat == 0.0 else (np.inf if diff_cat <= 0.0 else within_cat / diff_cat)

            q_con[i] = q_c
            q_cat[i] = q_k
            obj = q_c * q_k
            obj_fun[i] = obj

            if obj < best_obj:
                best_obj = obj
                best_w = w
                best_res = {
                    "labels": km.labels_,
                    "con_centers": con_centers,
                    "cat_centers": cat_centers,
                }

        if np.any(obj_fun == 0.0):
            raise ValueError("Zero distortion encountered in Modha-Spangler objective function.")

        self.weights_ = weights
        self.objective_values_ = obj_fun
        self.q_con_ = q_con
        self.q_cat_ = q_cat
        self.best_weight_ = best_w
        self.best_objective_ = best_obj
        self.labels_ = best_res["labels"]
        self.con_centers_ = best_res["con_centers"]
        self.cat_centers_ = best_res["cat_centers"]

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict the closest cluster for new samples.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ["labels_", "con_centers_", "cat_centers_"])

        X_con, X_cat, _, _, _, _ = extract_mixed_features(
            X,
            fit=False,
            categories=self.categorical_levels_,
            feature_names_in=getattr(self, "feature_names_in_", None),
            con_indices=self.con_indices_,
            cat_indices=self.cat_indices_,
        )

        X_cat_dum = self._dummy_code(X_cat, self.categorical_levels_)

        w = self.best_weight_
        con_scaled = X_con * w
        cat_scaled = X_cat_dum * (1.0 - w)
        X_combined = np.hstack([con_scaled, cat_scaled])

        combined_centers = np.hstack([self.con_centers_ * w, self.cat_centers_ * (1.0 - w)])

        # Closest centroid in weighted Euclidean space
        diff = X_combined[:, np.newaxis, :] - combined_centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        return np.argmin(dist_sq, axis=1)

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
        y : Ignored

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return self.fit(X, y).labels_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = False
        tags.input_tags.allow_nan = False
        return tags
