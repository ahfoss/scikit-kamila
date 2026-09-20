"""Core computational engine for KAMILA and Prediction Strength clustering."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from .utils._kde import binned_radial_kde


def dptm(
    pts: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute weighted Euclidean distance from points to centroids.

    Matches dptm from kamila C++ implementation:
    dist(x, mu) = sqrt( sum_p [ w_p * (x_p - mu_p) ]^2 )

    Parameters
    ----------
    pts : ndarray of shape (n_samples, n_features)
    means : ndarray of shape (n_clusters, n_features)
    weights : ndarray of shape (n_features,)

    Returns
    -------
    dist : ndarray of shape (n_samples, n_clusters)
    """
    pts = np.asarray(pts, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # pts: (N, 1, P), means: (1, K, P), weights: (1, 1, P)
    diff = (pts[:, np.newaxis, :] - means[np.newaxis, :, :]) * weights[np.newaxis, np.newaxis, :]
    dist_sq = np.sum(diff * diff, axis=2)
    return np.sqrt(dist_sq)


def calc_ps(
    test_memb: np.ndarray,
    te_into_tr: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Calculate cluster prediction strength using fast contingency partitioning.

    Matches calcPsCpp from kamila C++: O(N + K^2) time complexity.

    Parameters
    ----------
    test_memb : 1D ndarray of shape (n_test,) with values 0 .. n_clusters - 1
        Cluster memberships of test data clustered alone.
    te_into_tr : 1D ndarray of shape (n_test,) with values 0 .. n_pred - 1
        Predicted cluster memberships of test data classified using training model.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    ps_props : 1D ndarray of shape (n_clusters,)
        Prediction strength proportion for each cluster.
    """
    test_memb = np.asarray(test_memb, dtype=np.int32)
    te_into_tr = np.asarray(te_into_tr, dtype=np.int32)
    n = len(test_memb)

    if n == 0 or n_clusters <= 0:
        return np.full(n_clusters, np.nan)

    max_pred = int(np.max(te_into_tr)) + 1 if len(te_into_tr) > 0 else 0
    if max_pred == 0:
        return np.full(n_clusters, np.nan)

    # Contingency counts: counts[cl, tr]
    counts = np.zeros((n_clusters, max_pred), dtype=np.float64)
    clust_size = np.zeros(n_clusters, dtype=np.float64)

    for i in range(n):
        tm = test_memb[i]
        tr = te_into_tr[i]
        if 0 <= tm < n_clusters and 0 <= tr < max_pred:
            counts[tm, tr] += 1.0
            clust_size[tm] += 1.0

    ps_props = np.full(n_clusters, np.nan, dtype=np.float64)
    for cl in range(n_clusters):
        n_cl = clust_size[cl]
        if n_cl >= 2.0:
            # Number of agreeing pairs in cluster cl: sum_m C_{cl, m} * (C_{cl, m} - 1) / 2
            cnt = counts[cl, :]
            agreeing_pairs = np.sum(cnt * (cnt - 1.0)) / 2.0
            total_pairs = n_cl * (n_cl - 1.0) / 2.0
            ps_props[cl] = agreeing_pairs / total_pairs

    return ps_props


def update_cat_log_probs(
    X_cat: np.ndarray,
    membership: np.ndarray,
    num_lev: List[int],
    cat_bw: float,
    n_clusters: int,
) -> List[np.ndarray]:
    """Smooth and update categorical cluster-conditional log probabilities.

    Matches updateLogProbs from kamila C++ implementation.

    Parameters
    ----------
    X_cat : 2D ndarray of shape (n_samples, n_cat) with 0-based integer levels
    membership : 1D ndarray of shape (n_samples,) with 0-based cluster IDs
    num_lev : list of int of length n_cat
    cat_bw : float
        Categorical smoothing parameter.
    n_clusters : int

    Returns
    -------
    log_probs : list of 2D ndarrays of shape (n_clusters, n_levels)
    """
    qq = len(num_lev)
    log_probs_list = []

    for q in range(qq):
        nlev = num_lev[q]
        col_q = X_cat[:, q]

        # (a) Tabulate raw counts: raw_tab[cluster, level]
        raw_tab = np.zeros((n_clusters, nlev), dtype=np.float64)
        for i in range(len(col_q)):
            cl = membership[i]
            lev = col_q[i]
            if 0 <= cl < n_clusters and 0 <= lev < nlev:
                raw_tab[cl, lev] += 1.0

        # (b) Smoothing
        if cat_bw != 0.0:
            col_sums = np.sum(raw_tab, axis=0)  # shape (nlev,)
            bw_div_k = (cat_bw / (n_clusters - 1.0)) if n_clusters > 1 else 0.0
            one_minus_bw = 1.0 - cat_bw

            # mid_mat[i, j] = (1 - bw) * raw_tab[i, j] + bw_div_k * (col_sums[j] - raw_tab[i, j])
            mid_mat = (
                one_minus_bw * raw_tab + bw_div_k * (col_sums[np.newaxis, :] - raw_tab)
            )

            row_sums = np.sum(mid_mat, axis=1)  # shape (n_clusters,)
            bw_div_lev = (cat_bw / (nlev - 1.0)) if nlev > 1 else 0.0

            # out_mat[i, j] = (1 - bw) * mid_mat[i, j] + bw_div_lev * (row_sums[i] - mid_mat[i, j])
            out_mat = (
                one_minus_bw * mid_mat
                + bw_div_lev * (row_sums[:, np.newaxis] - mid_mat)
            )
        else:
            out_mat = raw_tab

        # (c) Normalize and take log
        final_row_sums = np.sum(out_mat, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = np.where(final_row_sums > 0.0, out_mat / final_row_sums, 0.0)
            log_p = np.where(probs > 0.0, np.log(probs), -np.inf)

        log_probs_list.append(log_p)

    return log_probs_list


def calc_cat_log_liks(
    X_cat: np.ndarray,
    cat_weights: np.ndarray,
    log_probs_list: List[np.ndarray],
    n_clusters: int,
) -> np.ndarray:
    """Calculate cluster categorical log-likelihoods for each observation.

    Parameters
    ----------
    X_cat : ndarray of shape (n_samples, n_cat)
    cat_weights : ndarray of shape (n_cat,)
    log_probs_list : list of ndarray, each of shape (n_clusters, n_levels)
    n_clusters : int

    Returns
    -------
    cat_log_liks : ndarray of shape (n_samples, n_clusters)
    """
    n_samples, qq = X_cat.shape
    cat_log_liks = np.zeros((n_samples, n_clusters), dtype=np.float64)

    for q in range(qq):
        w = cat_weights[q]
        if w == 0.0:
            continue
        col_q = X_cat[:, q]
        lp_q = log_probs_list[q]  # shape (n_clusters, n_levels)
        # lp_q[:, col_q] has shape (n_clusters, n_samples)
        cat_log_liks += w * lp_q[:, col_q].T

    return cat_log_liks


def kamila_loop(
    X_con: Optional[np.ndarray],
    X_cat: Optional[np.ndarray],
    con_weights: Optional[np.ndarray],
    cat_weights: Optional[np.ndarray],
    init_means: Optional[np.ndarray],
    init_log_probs: Optional[List[np.ndarray]],
    num_lev: List[int],
    cat_bw: float,
    n_clusters: int,
    max_iter: int,
    verbose: bool = False,
) -> Dict:
    """Run fused coordinate descent clustering loop for KAMILA.

    Parameters
    ----------
    X_con : ndarray of shape (n_samples, n_con) or None
    X_cat : ndarray of shape (n_samples, n_cat) or None
    con_weights : ndarray of shape (n_con,) or None
    cat_weights : ndarray of shape (n_cat,) or None
    init_means : ndarray of shape (n_clusters, n_con) or None
    init_log_probs : list of ndarray, each of shape (n_clusters, n_levels) or None
    num_lev : list of int
    cat_bw : float
    n_clusters : int
    max_iter : int
    verbose : bool

    Returns
    -------
    results : dict containing final_memb, final_means, final_log_probs, total_log_lik,
              cat_log_lik, win_dist, num_iter, degenerate_soln
    """
    has_con = X_con is not None
    has_cat = X_cat is not None
    n_samples = len(X_con) if has_con else len(X_cat)
    pp = X_con.shape[1] if has_con else 0
    qq = X_cat.shape[1] if has_cat else 0

    current_means = init_means.copy() if has_con and init_means is not None else None
    log_probs = [lp.copy() for lp in init_log_probs] if has_cat and init_log_probs is not None else None

    memb_old = np.zeros(n_samples, dtype=np.int32)
    memb_new = np.zeros(n_samples, dtype=np.int32)
    memb_history = []

    num_iter = 0
    degenerate_soln = False
    dist_mat = np.zeros((n_samples, n_clusters), dtype=np.float64) if has_con else None
    all_log_liks = np.zeros((n_samples, n_clusters), dtype=np.float64)
    cat_log_liks = np.zeros((n_samples, n_clusters), dtype=np.float64) if has_cat else None

    while True:
        # Check convergence after at least 3 iterations, matching R kamila
        if num_iter >= 3:
            if np.array_equal(memb_old, memb_new):
                break
        if num_iter >= max_iter:
            break

        num_iter += 1

        # 1. Continuous calculations: distances and radial KDE
        if has_con:
            dist_mat = dptm(X_con, current_means, con_weights)
            min_dist = np.min(dist_mat, axis=1)

            # Radial KDE evaluation
            log_rad_dens = binned_radial_kde(
                radii=min_dist,
                eval_points=dist_mat,
                pdim=pp,
                take_log=True,
            )
            all_log_liks = log_rad_dens.copy()
        else:
            all_log_liks.fill(0.0)

        # 2. Categorical calculations
        if has_cat:
            cat_log_liks = calc_cat_log_liks(X_cat, cat_weights, log_probs, n_clusters)
            if has_con:
                all_log_liks += cat_log_liks
            else:
                all_log_liks = cat_log_liks.copy()

        # 3. Partition data: update membership
        memb_old[:] = memb_new
        memb_new = np.argmax(all_log_liks, axis=1).astype(np.int32)

        # Count observations per cluster
        counts = np.bincount(memb_new, minlength=n_clusters)

        # 4. Update continuous means
        if has_con:
            for k in range(n_clusters):
                if counts[k] > 0:
                    current_means[k, :] = np.mean(X_con[memb_new == k], axis=0)

        # 5. Update categorical probabilities
        if has_cat:
            log_probs = update_cat_log_probs(
                X_cat=X_cat,
                membership=memb_new,
                num_lev=num_lev,
                cat_bw=cat_bw,
                n_clusters=n_clusters,
            )

        if verbose:
            memb_history.append(memb_old.copy())

        # Check degenerate solution
        if np.any(counts == 0):
            degenerate_soln = True
            break

    # Summary metrics
    if degenerate_soln:
        total_log_lik = -np.inf
    else:
        total_log_lik = float(np.sum(np.max(all_log_liks, axis=1)))

    cat_log_lik = float(np.sum(np.max(cat_log_liks, axis=1))) if has_cat else 0.0

    win_dist = 0.0
    if has_con and dist_mat is not None:
        win_dist = float(np.sum(dist_mat[np.arange(n_samples), memb_new]))

    return {
        "num_iter": num_iter,
        "degenerate_soln": degenerate_soln,
        "final_memb": memb_new,
        "final_means": current_means,
        "final_log_probs": log_probs,
        "total_log_lik": total_log_lik,
        "cat_log_lik": cat_log_lik,
        "win_dist": win_dist,
        "cat_log_liks": cat_log_liks if verbose and has_cat else None,
        "memb_history": memb_history if verbose else None,
    }
