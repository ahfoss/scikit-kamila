"""KAMILA: KAy-means for MIxed LArge datasets clustering."""

import numbers

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from . import _kamila_cpp
from ._validation import _validate_and_split_data

try:
    from sklearn.utils.validation import _check_feature_names
except ImportError:  # pragma: no cover  (scikit-learn < 1.6)

    def _check_feature_names(estimator, X, *, reset):
        return estimator._check_feature_names(X, reset=reset)


def _is_int(value):
    """Return True for integers, excluding bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_real(value):
    """Return True for real numbers, excluding bool."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


class KamilaClustering(ClusterMixin, BaseEstimator):
    r"""KAMILA clustering of mixed-type continuous and categorical data.

    KAMILA (KAy-means for MIxed LArge datasets) is an iterative clustering technique
    that balances continuous and categorical variable contributions dynamically using
    non-parametric radial kernel density estimation on continuous variables and
    multinomial log-likelihood smoothing on categorical variables.

    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to form.
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
        Ignored when ``init_means`` or ``init_log_probs`` is given; a single run
        from that initialization is performed instead.
    max_iter : int, default=25
        Maximum number of iterations of the KAMILA algorithm for a single
        run/initialization.
    cat_bandwidth : float, default=0.025
        Categorical smoothing parameter in [0, 1].
    con_weights : array-like of shape (n_con,), optional, default=None
        Weights for continuous features. If None, all receive weight 1.0.
    cat_weights : array-like of shape (n_cat,), optional, default=None
        Weights for categorical features. If None, all receive weight 1.0.
    random_state : int, RandomState instance, or None, default=None
        Determines random number generation for centroid initializations and
        for re-initializing clusters that become empty during fitting.
    init_means : array-like of shape (n_clusters, n_con), optional, default=None
        Explicit initial continuous cluster centers (for deterministic testing).
        Must be finite. For data with both continuous and categorical features,
        ``init_log_probs`` must be given as well.
    init_log_probs : list of array-like, optional, default=None
        Explicit initial categorical log-probability matrices, one of shape
        (n_clusters, n_levels) per categorical feature, with entries <= 0. For
        data with both continuous and categorical features, ``init_means`` must
        be given as well.

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
        Objective value of the selected initialization; higher is better (this
        is the opposite of scikit-learn's ``KMeans.inertia_``). For continuous-only
        data it equals ``total_log_lik_``; for categorical-only data it equals
        ``cat_log_lik_``; for mixed data it is the heuristic
        ``win_dist_ / (total_dist_ - win_dist_) * cat_log_lik_`` (the ratio falls
        back to 100 when it is undefined or negative), which is not a
        likelihood.
    n_iter_ : int
        Number of iterations run in the best initialization.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (n_features,)
        Names of features seen during :meth:`fit` (if X had feature names).
    is_categorical_ : ndarray of shape (n_features,) of bool
        Boolean mask indicating which features are categorical.
    categories_ : list of ndarray
        The categories / levels present in each categorical feature during :meth:`fit`.
    num_levels_ : ndarray of shape (n_cat,) of int32 or None
        Number of distinct levels for each categorical feature.
    total_log_lik_ : float or None
        Sum over training samples of the best per-cluster log-likelihood from the
        final iteration. For mixed data this combines the continuous radial
        kernel density term and the weighted categorical term; for
        continuous-only data it is the radial kernel density term alone. None if
        no continuous features.
    cat_log_lik_ : float or None
        Sum over training samples of the best per-cluster weighted categorical
        log-likelihood from the final iteration. None if no categorical features.
    win_dist_ : float or None
        Sum over training samples of the weighted Euclidean distance to the
        continuous center of the assigned cluster, measured with the centers
        from the start of the final iteration. None if no continuous features.
    total_dist_ : float
        Sum over training samples of the weighted Euclidean distance to the mean
        of the continuous features. 0.0 if no continuous features.
    fitted_min_dist_ : ndarray of shape (n_samples,) or None
        Minimum continuous distance from each training observation to fitted
        cluster centers.

    Notes
    -----
    If a cluster becomes empty during fitting, a randomly selected sample (drawn
    from clusters with more than one member) is moved into it, and the cluster's
    parameters are estimated from that membership. The selection is controlled by
    ``random_state``.
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
        if not _is_int(self.n_clusters) or self.n_clusters < 1:
            raise ValueError(
                f"n_clusters must be an integer >= 1; got {self.n_clusters!r}."
            )
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}."
            )
        if not _is_int(self.n_init) or self.n_init < 1:
            raise ValueError(f"n_init must be an integer >= 1; got {self.n_init!r}.")
        if not _is_int(self.max_iter) or self.max_iter < 1:
            raise ValueError(
                f"max_iter must be an integer >= 1; got {self.max_iter!r}."
            )
        if not _is_real(self.cat_bandwidth) or not 0 <= self.cat_bandwidth <= 1:
            raise ValueError(
                "cat_bandwidth must be a number in [0, 1]; got "
                f"{self.cat_bandwidth!r}."
            )

    def _validate_init(self, n_con, n_cat, num_levels):
        """Validate explicit initializations against the data.

        Returns contiguous float64 copies of ``init_means`` and ``init_log_probs``
        (None where not given).
        """
        has_means = self.init_means is not None
        has_log_probs = self.init_log_probs is not None

        if has_means and n_con == 0:
            raise ValueError("init_means was given but X has no continuous features.")
        if has_log_probs and n_cat == 0:
            raise ValueError(
                "init_log_probs was given but X has no categorical features."
            )
        if n_con > 0 and n_cat > 0 and has_means != has_log_probs:
            raise ValueError(
                "init_means and init_log_probs must both be given for data with "
                "continuous and categorical features."
            )

        init_means = None
        if has_means:
            init_means = np.ascontiguousarray(self.init_means, dtype=np.float64)
            expected = (self.n_clusters, n_con)
            if init_means.shape != expected:
                raise ValueError(
                    f"init_means must have shape {expected}; got "
                    f"{init_means.shape}."
                )
            if not np.all(np.isfinite(init_means)):
                raise ValueError("init_means must contain only finite values.")

        init_log_probs = None
        if has_log_probs:
            if len(self.init_log_probs) != n_cat:
                raise ValueError(
                    f"init_log_probs must contain one matrix per categorical "
                    f"feature ({n_cat}); got {len(self.init_log_probs)}."
                )
            init_log_probs = []
            for q, lp in enumerate(self.init_log_probs):
                lp = np.ascontiguousarray(lp, dtype=np.float64)
                expected = (self.n_clusters, int(num_levels[q]))
                if lp.shape != expected:
                    raise ValueError(
                        f"init_log_probs[{q}] must have shape {expected}; got "
                        f"{lp.shape}."
                    )
                if np.any(np.isnan(lp)) or np.any(lp > 0):
                    raise ValueError(
                        f"init_log_probs[{q}] must contain log-probabilities "
                        "(no NaN, all values <= 0)."
                    )
                init_log_probs.append(lp)

        return init_means, init_log_probs

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
            _,
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
        init_means_arr, init_lp_list = self._validate_init(n_con, n_cat, num_levels)

        self.n_features_in_ = n_features
        _check_feature_names(self, X, reset=True)
        self.is_categorical_ = is_categorical
        self.categories_ = categories
        self.num_levels_ = num_levels

        # Check explicit vs random initializations
        has_explicit_init = (init_means_arr is not None) or (init_lp_list is not None)
        rng = check_random_state(self.random_state)

        if has_explicit_init:
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
                seed=rng.randint(np.iinfo(np.int32).max),
            )
        else:
            best_res = None
            best_obj = -float("inf")

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
                    seed=rng.randint(np.iinfo(np.int32).max),
                )

                if best_res is None or (
                    not np.isnan(res["objective"]) and res["objective"] > best_obj
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
            self.fitted_min_dist_ = (
                np.asarray(best_res["final_min_dist"], dtype=np.float64)
                if "final_min_dist" in best_res
                else None
            )
        else:
            self.cluster_centers_con_ = None
            self.fitted_min_dist_ = None

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
        _check_feature_names(self, X, reset=False)

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

        min_dist_arr = (
            np.ascontiguousarray(self.fitted_min_dist_, dtype=np.float64)
            if self.fitted_min_dist_ is not None
            else None
        )

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
            all_data_min_dist=min_dist_arr,
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.input_tags.categorical = self.categorical_features is not None
        tags.array_api_support = False
        return tags

    def _more_tags(self):
        # Tags for scikit-learn < 1.6, which ignores __sklearn_tags__.
        X_types = ["2darray"]
        if self.categorical_features is not None:
            X_types.append("categorical")
        return {"X_types": X_types}
