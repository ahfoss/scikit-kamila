"""KAMILA: KAy-means for MIxed LArge datasets clustering."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

import math
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ._core_py import calc_cat_log_liks, calc_ps, dptm, kamila_loop
from .utils._kde import binned_radial_kde
from .utils._validation import extract_mixed_features


class KamilaClustering(BaseEstimator, ClusterMixin):
    """KAMILA clustering of mixed-type, continuous-only, or categorical-only data.

    KAMILA (KAy-means for MIxed LArge datasets) is an iterative clustering
    method that equitably balances the contribution of continuous and categorical
    variables without requiring dummy coding (Foss & Markatou, 2018).

    It models spherical clusters in the continuous domain using semi-parametric
    radial kernel density estimation (RKDE), and models categorical variables
    using smoothed multinomial distributions.

    Parameters
    ----------
    n_clusters : int or sequence of int, default=2
        Number of clusters to form. If calc_num_clusters='ps', this can be a sequence
        of integers (e.g. range(2, 6)) over which prediction strength is evaluated.
    n_init : int, default=10
        Number of times the algorithm is run with different centroid seeds.
        The final results are chosen based on the best objective score.
    max_iter : int, default=25
        Maximum number of iterations of the clustering algorithm for a single run.
    categorical_features : 'auto', list of column names/indices, boolean mask, or None, default=None
        Specifies which features are categorical.
        - None: all features are assumed continuous.
        - 'auto': columns with dtype 'category', 'object', or 'string' are treated as categorical.
        - list/array of column indices, column names, or boolean mask.
    con_weights : array-like of shape (n_continuous_features,), optional
        Continuous feature weights in [0, 1]. Defaults to uniform weights of 1.0.
    cat_weights : array-like of shape (n_categorical_features,), optional
        Categorical feature weights in [0, 1]. Defaults to uniform weights of 1.0.
    cat_bandwidth : float, default=0.025
        Bandwidth parameter used for categorical probability smoothing.
    con_init_method : {'runif', 'sample'}, default='runif'
        Method used to initialize continuous cluster means:
        - 'runif': Uniform random draws within feature bounding boxes.
        - 'sample': Random sample from observed feature values.
    calc_num_clusters : {'none', 'ps'}, default='none'
        Method for estimating the optimal number of clusters:
        - 'none': Fits model with the given n_clusters.
        - 'ps': Prediction strength cross-validation (Tibshirani & Walther, 2005).
    n_cv_runs : int, default=10
        Number of cross-validation runs when calc_num_clusters='ps'.
    pred_str_thresh : float, default=0.8
        Threshold for prediction strength cluster selection.
    n_jobs : int, optional
        Number of CPU cores to use for parallel execution across initializations and CV folds.
        None means 1 core. -1 means using all processors.
    random_state : int, RandomState instance, default=None
        Determines random number generation for initializations.
    verbose : bool or int, default=False
        Verbosity level.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    cluster_centers_ : ndarray of shape (n_clusters, n_con_features) or None
        Coordinates of cluster centers for continuous variables.
    categorical_probs_ : list of ndarray or None
        List of cluster-conditional level probabilities for each categorical variable.
    categorical_levels_ : list of ndarray or None
        Unique levels recorded for each categorical feature during fit.
    n_clusters_ : int
        Number of clusters in the fitted model.
    n_iter_ : int
        Number of iterations run in the best initialization.
    score_ : float
        Total pseudo log-likelihood of the selected partition.
    objective_value_ : float
        Objective value of the selected partition.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit (if X is a DataFrame).
    prediction_strength_results_ : dict or None
        Summary of prediction strength cross-validation if calc_num_clusters='ps'.

    References
    ----------
    .. [1] Foss A, Markatou M. "kamila: Clustering Mixed-Type Data in R and Hadoop."
           Journal of Statistical Software, 83(13), 1-45 (2018).
           doi: 10.18637/jss.v083.i13
    .. [2] Tibshirani R, Walther G. "Cluster Validation by Prediction Strength."
           Journal of Computational and Graphical Statistics, 14(3), 511-528 (2005).
    """

    def __init__(
        self,
        n_clusters: Union[int, Sequence[int]] = 2,
        n_init: int = 10,
        max_iter: int = 25,
        categorical_features: Optional[Union[str, Sequence[Union[int, str]], np.ndarray]] = None,
        con_weights: Optional[np.ndarray] = None,
        cat_weights: Optional[np.ndarray] = None,
        cat_bandwidth: float = 0.025,
        con_init_method: str = "runif",
        calc_num_clusters: str = "none",
        n_cv_runs: int = 10,
        pred_str_thresh: float = 0.8,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: Union[bool, int] = False,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.categorical_features = categorical_features
        self.con_weights = con_weights
        self.cat_weights = cat_weights
        self.cat_bandwidth = cat_bandwidth
        self.con_init_method = con_init_method
        self.calc_num_clusters = calc_num_clusters
        self.n_cv_runs = n_cv_runs
        self.pred_str_thresh = pred_str_thresh
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any = None) -> "KamilaClustering":
        """Compute KAMILA clustering.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate calc_num_clusters
        if self.calc_num_clusters not in ("none", "ps"):
            raise ValueError(
                f"calc_num_clusters must be 'none' or 'ps', got '{self.calc_num_clusters}'"
            )

        if self.con_init_method not in ("runif", "sample"):
            raise ValueError(
                f"con_init_method must be 'runif' or 'sample', got '{self.con_init_method}'"
            )

        if self.n_init <= 0:
            raise ValueError(f"n_init must be a positive integer, got {self.n_init}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be a positive integer, got {self.max_iter}")

        # Extract features
        X_con, X_cat, categories, con_idx, cat_idx, feat_names = extract_mixed_features(
            X, categorical_features=self.categorical_features, fit=True
        )

        n_samples = len(X_con) if X_con is not None else len(X_cat)
        has_con = X_con is not None
        has_cat = X_cat is not None
        n_con = X_con.shape[1] if has_con else 0
        n_cat = X_cat.shape[1] if has_cat else 0

        # Validate weights
        con_weights_arr = None
        if has_con:
            if self.con_weights is None:
                con_weights_arr = np.ones(n_con, dtype=np.float64)
            else:
                con_weights_arr = np.asarray(self.con_weights, dtype=np.float64)
                if len(con_weights_arr) != n_con:
                    raise ValueError(
                        f"con_weights length ({len(con_weights_arr)}) must match continuous features ({n_con})"
                    )
                if np.min(con_weights_arr) < 0.0 or np.max(con_weights_arr) > 1.0:
                    raise ValueError("con_weights must be in [0, 1]")

        cat_weights_arr = None
        if has_cat:
            if self.cat_weights is None:
                cat_weights_arr = np.ones(n_cat, dtype=np.float64)
            else:
                cat_weights_arr = np.asarray(self.cat_weights, dtype=np.float64)
                if len(cat_weights_arr) != n_cat:
                    raise ValueError(
                        f"cat_weights length ({len(cat_weights_arr)}) must match categorical features ({n_cat})"
                    )
                if np.min(cat_weights_arr) < 0.0 or np.max(cat_weights_arr) > 1.0:
                    raise ValueError("cat_weights must be in [0, 1]")

        num_lev = [len(c) for c in categories] if has_cat else []

        # Store metadata attributes
        self.n_features_in_ = (n_con + n_cat)
        if feat_names is not None:
            self.feature_names_in_ = feat_names
        self.con_indices_ = con_idx
        self.cat_indices_ = cat_idx
        self.categorical_levels_ = categories

        # Check prediction strength mode
        if self.calc_num_clusters == "ps":
            if isinstance(self.n_clusters, (int, np.integer)):
                candidates = [int(self.n_clusters)]
            else:
                candidates = list(self.n_clusters)

            if len(candidates) < 1:
                raise ValueError("n_clusters must contain at least one candidate.")

            best_k, ps_summary = self._fit_prediction_strength(
                X_con=X_con,
                X_cat=X_cat,
                con_weights=con_weights_arr,
                cat_weights=cat_weights_arr,
                num_lev=num_lev,
                candidates=candidates,
            )
            self.prediction_strength_results_ = ps_summary
            chosen_k = best_k
        else:
            if not isinstance(self.n_clusters, (int, np.integer)):
                raise ValueError(
                    "n_clusters must be an integer when calc_num_clusters='none'"
                )
            chosen_k = int(self.n_clusters)
            self.prediction_strength_results_ = None

        if chosen_k < 1:
            raise ValueError(f"n_clusters must be >= 1, got {chosen_k}")
        if chosen_k > n_samples:
            raise ValueError(
                f"n_clusters ({chosen_k}) cannot exceed number of samples ({n_samples})"
            )

        # Run final fit with chosen_k
        fit_res = self._fit_single_k(
            X_con=X_con,
            X_cat=X_cat,
            con_weights=con_weights_arr,
            cat_weights=cat_weights_arr,
            num_lev=num_lev,
            k=chosen_k,
            n_init=self.n_init,
            random_state=self.random_state,
        )

        self.n_clusters_ = chosen_k
        self.labels_ = fit_res["labels"]
        self.cluster_centers_ = fit_res["centers"]
        self.categorical_probs_ = fit_res["cat_probs"]
        self.categorical_log_probs_ = fit_res["cat_log_probs"]
        self.n_iter_ = fit_res["n_iter"]
        self.score_ = fit_res["total_log_lik"]
        self.objective_value_ = fit_res["objective"]
        self.con_weights_ = con_weights_arr
        self.cat_weights_ = cat_weights_arr

        return self

    def _init_means(self, X_con: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
        """Initialize continuous means."""
        n_samples, pp = X_con.shape
        means = np.empty((k, pp), dtype=np.float64)
        if self.con_init_method == "sample":
            for j in range(pp):
                means[:, j] = rng.choice(X_con[:, j], size=k, replace=True)
        elif self.con_init_method == "runif":
            for j in range(pp):
                min_v = float(np.min(X_con[:, j]))
                max_v = float(np.max(X_con[:, j]))
                means[:, j] = rng.uniform(low=min_v, high=max_v, size=k)
        return means

    def _init_cat_log_probs(
        self, num_lev: List[int], k: int, rng: np.random.RandomState
    ) -> List[np.ndarray]:
        """Initialize categorical log probabilities using Dirichlet(1)."""
        log_probs = []
        for nlev in num_lev:
            probs = rng.dirichlet(alpha=np.ones(nlev), size=k)
            log_probs.append(np.log(probs))
        return log_probs

    def _fit_single_k(
        self,
        X_con: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        con_weights: Optional[np.ndarray],
        cat_weights: Optional[np.ndarray],
        num_lev: List[int],
        k: int,
        n_init: int,
        random_state: Any,
    ) -> Dict:
        """Execute multiple initializations for a fixed k and pick the best."""
        rng = check_random_state(random_state)
        has_con = X_con is not None
        has_cat = X_cat is not None
        n_samples = len(X_con) if has_con else len(X_cat)

        total_dist = 0.0
        if has_con:
            overall_mean = np.mean(X_con, axis=0, keepdims=True)
            dist_to_mean = dptm(X_con, overall_mean, con_weights)
            total_dist = float(np.sum(dist_to_mean))

        seeds = rng.randint(0, np.iinfo(np.int32).max, size=n_init)

        def _run_single_init(init_seed):
            init_rng = np.random.RandomState(init_seed)
            init_means = self._init_means(X_con, k, init_rng) if has_con else None
            init_lp = self._init_cat_log_probs(num_lev, k, init_rng) if has_cat else None

            res = kamila_loop(
                X_con=X_con,
                X_cat=X_cat,
                con_weights=con_weights,
                cat_weights=cat_weights,
                init_means=init_means,
                init_log_probs=init_lp,
                num_lev=num_lev,
                cat_bw=self.cat_bandwidth,
                n_clusters=k,
                max_iter=self.max_iter,
                verbose=False,
            )

            # Compute objective matching R kamila
            if has_con and has_cat:
                win_dist = res["win_dist"]
                denom = total_dist - win_dist
                ratio = (win_dist / denom) if denom > 0.0 else 100.0
                if ratio < 0.0:
                    ratio = 100.0
                objective = ratio * res["cat_log_lik"]
            elif has_con:
                objective = res["total_log_lik"]
            else:
                objective = -np.inf if res["degenerate_soln"] else res["cat_log_lik"]

            return objective, res

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_single_init)(seed) for seed in seeds
        )

        best_obj = -np.inf
        best_res = None
        for obj, res in results:
            if res["degenerate_soln"]:
                continue
            if obj > best_obj or best_res is None:
                best_obj = obj
                best_res = res

        if best_res is None:
            # Fallback if all degenerate
            best_res = results[0][1]
            best_obj = results[0][0]

        cat_probs = None
        cat_log_probs = None
        if has_cat and best_res["final_log_probs"] is not None:
            cat_log_probs = best_res["final_log_probs"]
            cat_probs = [np.exp(lp) for lp in cat_log_probs]

        return {
            "labels": best_res["final_memb"],
            "centers": best_res["final_means"],
            "cat_probs": cat_probs,
            "cat_log_probs": cat_log_probs,
            "n_iter": best_res["num_iter"],
            "total_log_lik": best_res["total_log_lik"],
            "objective": best_obj,
        }

    def _fit_prediction_strength(
        self,
        X_con: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        con_weights: Optional[np.ndarray],
        cat_weights: Optional[np.ndarray],
        num_lev: List[int],
        candidates: List[int],
    ) -> Tuple[int, Dict]:
        """Estimate optimal number of clusters via prediction strength cross-validation."""
        rng = check_random_state(self.random_state)
        n_samples = len(X_con) if X_con is not None else len(X_cat)
        has_con = X_con is not None
        has_cat = X_cat is not None

        ps_scores_mean = {}
        ps_scores_se = {}
        n_half = n_samples // 2

        sorted_candidates = sorted(candidates)

        for k in sorted_candidates:
            fold_ps = []
            for cv in range(self.n_cv_runs):
                perm = rng.permutation(n_samples)
                train_idx, test_idx = perm[:n_half], perm[n_half:]

                # Subset training and test
                tr_con = X_con[train_idx] if has_con else None
                tr_cat = X_cat[train_idx] if has_cat else None
                te_con = X_con[test_idx] if has_con else None
                te_cat = X_cat[test_idx] if has_cat else None

                # 1. Fit on training set
                tr_fit = self._fit_single_k(
                    X_con=tr_con,
                    X_cat=tr_cat,
                    con_weights=con_weights,
                    cat_weights=cat_weights,
                    num_lev=num_lev,
                    k=k,
                    n_init=self.n_init,
                    random_state=rng.randint(0, np.iinfo(np.int32).max),
                )

                # 2. Fit on test set independently
                te_fit = self._fit_single_k(
                    X_con=te_con,
                    X_cat=te_cat,
                    con_weights=con_weights,
                    cat_weights=cat_weights,
                    num_lev=num_lev,
                    k=k,
                    n_init=self.n_init,
                    random_state=rng.randint(0, np.iinfo(np.int32).max),
                )

                # 3. Classify test into training clusters
                te_into_tr = self._classify(
                    X_con=te_con,
                    X_cat=te_cat,
                    centers=tr_fit["centers"],
                    cat_log_probs=tr_fit["cat_log_probs"],
                    con_weights=con_weights,
                    cat_weights=cat_weights,
                    k=k,
                )

                # 4. Calculate prediction strength
                ps_per_cluster = calc_ps(
                    test_memb=te_fit["labels"],
                    te_into_tr=te_into_tr,
                    n_clusters=k,
                )
                min_ps = float(np.nanmin(ps_per_cluster))
                fold_ps.append(min_ps)

            fold_arr = np.array(fold_ps)
            valid = fold_arr[~np.isnan(fold_arr)]
            if len(valid) > 0:
                mean_ps = float(np.mean(valid))
                se_ps = float(np.std(valid, ddof=1) / math.sqrt(len(valid))) if len(valid) > 1 else 0.0
            else:
                mean_ps = 0.0
                se_ps = 0.0

            ps_scores_mean[k] = mean_ps
            ps_scores_se[k] = se_ps

        # Select largest k whose mean (or mean + se) >= threshold
        # Tibshirani & Walther (2005): largest k such that score >= thresh
        chosen_k = sorted_candidates[0]
        for k in sorted_candidates:
            if ps_scores_mean[k] >= self.pred_str_thresh:
                chosen_k = k

        summary = {
            "candidates": sorted_candidates,
            "mean_ps": ps_scores_mean,
            "se_ps": ps_scores_se,
            "best_k": chosen_k,
        }
        return chosen_k, summary

    def _classify(
        self,
        X_con: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        centers: Optional[np.ndarray],
        cat_log_probs: Optional[List[np.ndarray]],
        con_weights: Optional[np.ndarray],
        cat_weights: Optional[np.ndarray],
        k: int,
    ) -> np.ndarray:
        """Assign observations to closest cluster based on joint log-likelihood."""
        has_con = X_con is not None and centers is not None
        has_cat = X_cat is not None and cat_log_probs is not None
        n_samples = len(X_con) if has_con else len(X_cat)

        all_log_liks = np.zeros((n_samples, k), dtype=np.float64)

        if has_con:
            dist_mat = dptm(X_con, centers, con_weights)
            min_dist = np.min(dist_mat, axis=1)
            pp = X_con.shape[1]
            log_rad_dens = binned_radial_kde(
                radii=min_dist,
                eval_points=dist_mat,
                pdim=pp,
                take_log=True,
            )
            all_log_liks += log_rad_dens

        if has_cat:
            cat_log_liks = calc_cat_log_liks(X_cat, cat_weights, cat_log_probs, k)
            all_log_liks += cat_log_liks

        return np.argmax(all_log_liks, axis=1).astype(np.int32)

    def predict(self, X: Any) -> np.ndarray:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, ["labels_", "n_clusters_"])

        X_con, X_cat, _, _, _, _ = extract_mixed_features(
            X,
            fit=False,
            categories=self.categorical_levels_,
            feature_names_in=getattr(self, "feature_names_in_", None),
            con_indices=self.con_indices_,
            cat_indices=self.cat_indices_,
        )

        return self._classify(
            X_con=X_con,
            X_cat=X_cat,
            centers=self.cluster_centers_,
            cat_log_probs=self.categorical_log_probs_,
            con_weights=self.con_weights_,
            cat_weights=self.cat_weights_,
            k=self.n_clusters_,
        )

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X).labels_.

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
