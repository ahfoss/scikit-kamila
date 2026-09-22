"""KAMILA: KAy-means for MIxed LArge datasets clustering."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from . import _kamila_cpp
from ._validation import _validate_and_split_data


class KamilaClustering(BaseEstimator, ClusterMixin):
    r"""KAMILA clustering of mixed-type continuous and categorical data.

    KAMILA (KAy-means for MIxed LArge datasets) is an iterative clustering technique
    that balances continuous and categorical variable contributions dynamically using
    non-parametric radial kernel density estimation on continuous variables and
    multinomial log-likelihood smoothing on categorical variables.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form as well as the number of centroids to generate.
    categorical_features : None, 'from_dtype', or array-like, default=None
        Indicates which features are categorical:
        - ``None``: all features are treated as continuous.
        - ``'from_dtype'``: features with pandas ``category`` or ``bool`` dtypes.
        - array-like of int: integer indices of categorical columns.
        - array-like of str: names of categorical columns (when X has column names).
        - array-like of bool: boolean mask of shape (n_features,).
    n_init : int, default=10
        Number of times the algorithm will be run with different centroid seeds.
        The final results will be the best output of n_init runs in terms of objective.
    max_iter : int, default=25
        Maximum number of iterations of the KAMILA algorithm for a single run.
    cat_bandwidth : float, default=0.025
        Categorical smoothing parameter between 0 and 1.
    con_weights : array-like of shape (n_con,), optional, default=None
        Weights for continuous features. If None, all receive weight 1.0.
    cat_weights : array-like of shape (n_cat,), optional, default=None
        Weights for categorical features. If None, all receive weight 1.0.
    random_state : int, RandomState instance, or None, default=None
        Determines random number generation for centroid initializations.
    init_means : array-like of shape (n_clusters, n_con), optional, default=None
        Explicit initial continuous cluster centers (for deterministic testing).
    init_log_probs : list of array-like, optional, default=None
        Explicit initial categorical log-probability matrices.

    Attributes
    ----------
    cluster_centers_con_ : ndarray of shape (n_clusters, n_con) or None
        Coordinates of continuous cluster centers. None if no continuous features.
    cluster_centers_cat_ : list of ndarray or None
        List of categorical log-probability matrices for each categorical feature.
        None if no categorical features are present.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point in the training set.
    inertia_ : float
        Sum of squared distances / objective value of the best initialization.
    n_iter_ : int
        Number of iterations run in the best initialization.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during :meth:`fit` (if X had feature names).
    is_categorical_ : ndarray of shape (n_features_in_,) of bool
        Boolean mask indicating which features are categorical.
    categories_ : list of ndarray
        The categories / levels present in each categorical feature during :meth:`fit`.
    num_levels_ : ndarray of shape (n_cat,) of int32 or None
        Number of distinct levels for each categorical feature.
    total_log_lik_ : float or None
        Continuous radial kernel density log-likelihood of the fitted solution.
    cat_log_lik_ : float or None
        Categorical log-likelihood of the fitted solution.
    win_dist_ : float or None
        Minimum continuous distance summary statistic.
    total_dist_ : float or None
        Total combined distance / objective value.
    """

    def __init__(
        self,
        n_clusters=2,
        *,
        categorical_features=None,
        n_init=10,
        max_iter=25,
        cat_bandwidth=0.025,
        con_weights=None,
        cat_weights=None,
        random_state=None,
        init_means=None,
        init_log_probs=None,
    ):
        self.n_clusters = n_clusters
        self.categorical_features = categorical_features
        self.n_init = n_init
        self.max_iter = max_iter
        self.cat_bandwidth = cat_bandwidth
        self.con_weights = con_weights
        self.cat_weights = cat_weights
        self.random_state = random_state
        self.init_means = init_means
        self.init_log_probs = init_log_probs

    def _validate_parameters(self, n_samples):
        """Validate estimator hyperparameters."""
        if not isinstance(self.n_clusters, (int, np.integer)) or self.n_clusters < 1:
            raise ValueError(
                f"n_clusters must be an integer >= 1; got {self.n_clusters!r}."
            )
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}."
            )
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError(f"n_init must be an integer >= 1; got {self.n_init!r}.")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError(
                f"max_iter must be an integer >= 1; got {self.max_iter!r}."
            )
        if (
            not isinstance(self.cat_bandwidth, (int, float, np.number))
            or self.cat_bandwidth < 0
        ):
            raise ValueError(
                "cat_bandwidth must be a non-negative number; got "
                f"{self.cat_bandwidth!r}."
            )

    def fit(self, X, y=None):
        """Compute KAMILA clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Determine shape and feature names
        (
            X_con,
            X_cat,
            is_categorical,
            categories,
            num_levels,
            con_wgts,
            cat_wgts,
            feature_names,
        ) = _validate_and_split_data(
            X,
            categorical_features=self.categorical_features,
            con_weights=self.con_weights,
            cat_weights=self.cat_weights,
        )

        n_samples = X_con.shape[0] if X_con is not None else X_cat.shape[0]
        n_features = len(is_categorical)
        n_con = X_con.shape[1] if X_con is not None else 0
        n_cat = X_cat.shape[1] if X_cat is not None else 0

        self._validate_parameters(n_samples)

        self.n_features_in_ = n_features
        if feature_names is not None:
            self.feature_names_in_ = feature_names
        self.is_categorical_ = is_categorical
        self.categories_ = categories
        self.num_levels_ = num_levels

        # Check explicit vs random initializations
        has_explicit_init = (self.init_means is not None) or (
            self.init_log_probs is not None
        )

        if has_explicit_init:
            init_means_arr = (
                np.ascontiguousarray(self.init_means, dtype=np.float64)
                if self.init_means is not None
                else None
            )
            init_lp_list = (
                [
                    np.ascontiguousarray(lp, dtype=np.float64)
                    for lp in self.init_log_probs
                ]
                if self.init_log_probs is not None
                else None
            )
            best_res = _kamila_cpp.kamila_loop_cpp(
                con_data=X_con,
                cat_data=X_cat,
                n_samples=n_samples,
                n_con=n_con,
                n_cat=n_cat,
                n_clusters=int(self.n_clusters),
                con_weights=con_wgts,
                cat_weights=cat_wgts,
                num_levels=num_levels,
                init_means=init_means_arr,
                init_log_probs=init_lp_list,
                cat_bw=float(self.cat_bandwidth),
                max_iter=int(self.max_iter),
                has_con=bool(n_con > 0),
                has_cat=bool(n_cat > 0),
            )
        else:
            rng = check_random_state(self.random_state)
            best_res = None
            best_obj = float("inf")

            for _ in range(self.n_init):
                if n_con > 0:
                    con_min = np.min(X_con, axis=0)
                    con_max = np.max(X_con, axis=0)
                    # Sample uniform initial means within continuous bounding box
                    init_m = rng.uniform(
                        con_min, con_max, size=(self.n_clusters, n_con)
                    )
                    init_m = np.ascontiguousarray(init_m, dtype=np.float64)
                else:
                    init_m = None

                if n_cat > 0:
                    init_lp = []
                    for j in range(n_cat):
                        # Symmetric Dirichlet draw for category probabilities
                        p = rng.dirichlet(np.ones(num_levels[j]), size=self.n_clusters)
                        init_lp.append(
                            np.ascontiguousarray(np.log(p), dtype=np.float64)
                        )
                else:
                    init_lp = None

                res = _kamila_cpp.kamila_loop_cpp(
                    con_data=X_con,
                    cat_data=X_cat,
                    n_samples=n_samples,
                    n_con=n_con,
                    n_cat=n_cat,
                    n_clusters=int(self.n_clusters),
                    con_weights=con_wgts,
                    cat_weights=cat_wgts,
                    num_levels=num_levels,
                    init_means=init_m,
                    init_log_probs=init_lp,
                    cat_bw=float(self.cat_bandwidth),
                    max_iter=int(self.max_iter),
                    has_con=bool(n_con > 0),
                    has_cat=bool(n_cat > 0),
                )

                if best_res is None or (
                    not np.isnan(res["objective"]) and res["objective"] < best_obj
                ):
                    best_obj = res["objective"]
                    best_res = res

        # Unpack best result
        self.labels_ = np.asarray(best_res["final_membership"], dtype=np.int32)
        self.n_iter_ = int(best_res["num_iter"])
        self.inertia_ = float(best_res["objective"])
        self.win_dist_ = float(best_res["win_dist"]) if n_con > 0 else None
        self.total_dist_ = float(best_res["total_dist"])
        self.total_log_lik_ = float(best_res["total_log_lik"]) if n_con > 0 else None
        self.cat_log_lik_ = float(best_res["cat_log_lik"]) if n_cat > 0 else None

        if n_con > 0:
            self.cluster_centers_con_ = np.asarray(
                best_res["final_means"], dtype=np.float64
            ).reshape((self.n_clusters, n_con))
        else:
            self.cluster_centers_con_ = None

        if n_cat > 0:
            self.cluster_centers_cat_ = [
                np.asarray(lp, dtype=np.float64).reshape(
                    (self.n_clusters, num_levels[j])
                )
                for j, lp in enumerate(best_res["final_log_probs"])
            ]
        else:
            self.cluster_centers_cat_ = None

        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,) of int32
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, attributes=["labels_"])

        (
            X_con,
            X_cat,
            _,
            _,
            _,
            con_wgts,
            cat_wgts,
            _,
        ) = _validate_and_split_data(
            X,
            categories=self.categories_,
            con_weights=self.con_weights,
            cat_weights=self.cat_weights,
            is_categorical=self.is_categorical_,
        )

        n_samples = X_con.shape[0] if X_con is not None else X_cat.shape[0]
        n_con = X_con.shape[1] if X_con is not None else 0
        n_cat = X_cat.shape[1] if X_cat is not None else 0

        preds = _kamila_cpp.kamila_predict_cpp(
            con_data=X_con,
            cat_data=X_cat,
            n_samples=n_samples,
            n_con=n_con,
            n_cat=n_cat,
            n_clusters=int(self.n_clusters),
            con_weights=con_wgts,
            cat_weights=cat_wgts,
            fitted_means=self.cluster_centers_con_,
            fitted_log_probs=self.cluster_centers_cat_,
            has_con=bool(n_con > 0),
            has_cat=bool(n_cat > 0),
        )
        return np.asarray(preds, dtype=np.int32)

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by predicting on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,) of int32
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_
