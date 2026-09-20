"""Radial kernel density estimation and bandwidth selection.

This module reproduces the 1D binned Gaussian kernel density estimation and radial
Jacobian transformation used in the KAMILA algorithm (Foss & Markatou, 2018).
"""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

import math
import numpy as np


def bw_nrd0(x: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth selector matching R's stats::bw.nrd0.

    Parameters
    ----------
    x : 1D ndarray
        Sample data.

    Returns
    -------
    h : float
        Selected bandwidth.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 1.0

    # Sample standard deviation (ddof=1)
    std_val = float(np.std(x, ddof=1))

    # Type 7 quantiles for IQR, matching R's default quantile()
    q25, q75 = np.percentile(x, [25.0, 75.0])
    iqr_val = (q75 - q25) / 1.34

    lo = min(std_val, iqr_val) if iqr_val > 0.0 else std_val
    if lo <= 0.0 or np.isnan(lo):
        lo = std_val
        if lo <= 0.0 or np.isnan(lo):
            lo = abs(x[0]) if abs(x[0]) > 0.0 else 1.0

    return float(0.9 * lo * (n ** (-0.2)))


def interp_radial_kde(
    y_dens: np.ndarray,
    max_eval: float,
    pdim: int,
    eval_points: np.ndarray,
    take_log: bool = False,
    m_grid: int = 401,
) -> np.ndarray:
    """Evaluate radial KDE via Jacobian transformation and linear interpolation.

    Matches interpRadialKde from kamila C++ implementation.

    Parameters
    ----------
    y_dens : 1D ndarray of shape (m_grid,)
        Density values on the regular grid [0, max_eval].
    max_eval : float
        Maximum radius evaluation boundary.
    pdim : int
        Continuous dimensionality.
    eval_points : ndarray
        Points (radii) at which to evaluate density.
    take_log : bool, default=False
        Whether to return natural logarithm of densities.
    m_grid : int, default=401
        Number of grid points.

    Returns
    -------
    kdes : ndarray of same shape as eval_points
    """
    eval_points = np.asarray(eval_points, dtype=np.float64)
    orig_shape = eval_points.shape
    eval_flat = eval_points.ravel()
    n_eval = len(eval_flat)

    h = (max_eval / (m_grid - 1.0)) if max_eval > 0.0 else 1.0
    x_grid = np.arange(m_grid, dtype=np.float64) * h

    # 1. Remove non-positive estimates
    new_y = y_dens.copy()
    pos_mask = new_y > 0.0
    min_pos = np.min(new_y[pos_mask]) if np.any(pos_mask) else 1e-10
    new_y[~pos_mask] = min_pos / 100.0

    # 2. Bottom 5th percentile linear replacement (index 19 in 0-based indexing for 401 points)
    if x_grid[19] > 0.0:
        slope = new_y[19] / x_grid[19]
    else:
        slope = 0.0
    new_y[:20] = x_grid[:20] * slope

    # 3. Radial Jacobian transformation: divide by r^(pdim - 1)
    rad_y = np.empty(m_grid, dtype=np.float64)
    if pdim > 1:
        rad_y[1:] = new_y[1:] / (x_grid[1:] ** (pdim - 1))
    else:
        rad_y[1:] = new_y[1:]
    rad_y[0] = rad_y[1]

    # 4. Truncate at MAXDENS = 1.0
    np.minimum(rad_y, 1.0, out=rad_y)

    # 5. Normalize area to 1.0
    sum_rad_y = float(np.sum(rad_y))
    norm_factor = h * sum_rad_y
    if norm_factor > 0.0:
        dens_r = rad_y / norm_factor
    else:
        dens_r = np.zeros_like(rad_y)

    min_dens_r = float(np.min(dens_r))
    diff_dens_r = np.diff(dens_r)

    # 6. Linear interpolation
    inv_h = (1.0 / h) if h > 0.0 else 0.0
    kdes = np.empty(n_eval, dtype=np.float64)

    val0 = dens_r[0] if dens_r[0] > min_dens_r else min_dens_r
    val_max = dens_r[-1] if dens_r[-1] > min_dens_r else min_dens_r

    if take_log:
        log_val0 = math.log(val0)
        log_val_max = math.log(val_max)
        for i in range(n_eval):
            u = eval_flat[i]
            if u <= 0.0:
                kdes[i] = log_val0
            elif u >= max_eval:
                kdes[i] = log_val_max
            else:
                pos = u * inv_h
                idx = int(pos)
                if idx >= m_grid - 1:
                    idx = m_grid - 2
                frac = pos - idx
                val = dens_r[idx] + frac * diff_dens_r[idx]
                kdes[i] = math.log(val if val > min_dens_r else min_dens_r)
    else:
        for i in range(n_eval):
            u = eval_flat[i]
            if u <= 0.0:
                kdes[i] = val0
            elif u >= max_eval:
                kdes[i] = val_max
            else:
                pos = u * inv_h
                idx = int(pos)
                if idx >= m_grid - 1:
                    idx = m_grid - 2
                frac = pos - idx
                val = dens_r[idx] + frac * diff_dens_r[idx]
                kdes[i] = val if val > min_dens_r else min_dens_r

    return kdes.reshape(orig_shape)


def binned_radial_kde(
    radii: np.ndarray,
    eval_points: np.ndarray,
    pdim: int,
    take_log: bool = False,
    m_grid: int = 401,
) -> np.ndarray:
    """1D Binned Gaussian KDE and Radial transformation matching R kamila.

    Parameters
    ----------
    radii : 1D ndarray
        Distances from observations to their assigned cluster centers.
    eval_points : ndarray
        Distances at which to evaluate the radial density.
    pdim : int
        Number of continuous dimensions.
    take_log : bool, default=False
        Whether to return log densities.
    m_grid : int, default=401
        Grid size for binned KDE.

    Returns
    -------
    log_dens : ndarray of same shape as eval_points
    """
    radii = np.asarray(radii, dtype=np.float64)
    eval_points = np.asarray(eval_points, dtype=np.float64)
    n = len(radii)

    max_eval = float(np.max(eval_points)) if eval_points.size > 0 else 1.0
    if max_eval <= 0.0:
        max_eval = 1.0

    # 1. Bandwidth selection
    h_bw = bw_nrd0(radii)

    # 2. 1D Linear binning onto [0, max_eval]
    delta_grid = max_eval / (m_grid - 1.0)
    inv_delta = 1.0 / delta_grid
    gcounts = np.zeros(m_grid, dtype=np.float64)

    for r in radii:
        if 0.0 <= r < max_eval:
            pos = r * inv_delta
            l = int(pos)
            rem = pos - l
            if 0 <= l < m_grid - 1:
                gcounts[l] += 1.0 - rem
                gcounts[l + 1] += rem

    # 3. Discrete Gaussian convolution (KernSmooth / bkde style)
    delta = delta_grid / h_bw
    L = int(math.floor(4.0 / delta)) if delta > 0.0 else m_grid
    if L > m_grid:
        L = m_grid
    if L < 0:
        L = 0

    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    l_indices = np.arange(L + 1, dtype=np.float64)
    z = l_indices * delta
    kappa = np.exp(-0.5 * z * z) * (inv_sqrt_2pi / (n * h_bw))

    sum_kappa = kappa[0] + 2.0 * np.sum(kappa[1:])
    tot = sum_kappa * delta_grid * n
    if tot > 0.0:
        kappa /= tot

    # Convolve gcounts with symmetric kernel kappa
    y_kde = np.zeros(m_grid, dtype=np.float64)
    for i in range(m_grid):
        j_min = max(0, i - L)
        j_max = min(m_grid - 1, i + L)
        diffs = np.abs(i - np.arange(j_min, j_max + 1))
        y_kde[i] = np.dot(gcounts[j_min : j_max + 1], kappa[diffs])

    # 4. Interpolate and transform
    return interp_radial_kde(
        y_dens=y_kde,
        max_eval=max_eval,
        pdim=pdim,
        eval_points=eval_points,
        take_log=take_log,
        m_grid=m_grid,
    )
