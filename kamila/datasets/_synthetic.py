"""Synthetic dataset generators for mixed-type clustering."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import norm
from sklearn.utils import check_random_state


def make_mixed_data(
    n_samples: int = 200,
    n_con_features: int = 2,
    n_cat_features: int = 2,
    n_cat_levels: int = 4,
    n_con_with_err: int = 1,
    n_cat_with_err: int = 1,
    pop_proportions: Tuple[float, float] = (0.5, 0.5),
    con_err_level: float = 0.2,
    cat_err_level: float = 0.2,
    random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
    return_dataframe: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[Any, np.ndarray]]:
    """Generate synthetic mixed continuous and categorical data with latent cluster structure.

    Simulates mixed-type datasets with two latent populations, matching the genMixedData
    function from the kamila R package (Foss & Markatou, 2018).

    Parameters
    ----------
    n_samples : int, default=200
        Number of samples to generate.
    n_con_features : int, default=2
        Number of continuous variables.
    n_cat_features : int, default=2
        Number of categorical variables.
    n_cat_levels : int, default=4
        Number of levels per categorical variable. Must be an even number.
    n_con_with_err : int, default=1
        Number of continuous features with measurement error (noise/overlap).
    n_cat_with_err : int, default=1
        Number of categorical features with measurement error (noise/overlap).
    pop_proportions : tuple of float, default=(0.5, 0.5)
        Prior population mixture weights (must sum to 1).
    con_err_level : float, default=0.2
        Overlap level in [0.01, 1.0] for continuous error variables.
    cat_err_level : float, default=0.2
        Overlap level in [0.01, 1.0] for categorical error variables.
    random_state : int, RandomState or None, default=None
        Determines random number generation.
    return_dataframe : bool, default=False
        If True, returns a pandas DataFrame with continuous and categorical columns,
        and an array of ground truth labels `(df, y)`.
        If False, returns `(X_con, X_cat, y)`.

    Returns
    -------
    X_con : ndarray of shape (n_samples, n_con_features)
        Continuous variables.
    X_cat : ndarray of shape (n_samples, n_cat_features)
        Categorical variables (0-indexed integers).
    y : ndarray of shape (n_samples,)
        Ground-truth cluster assignments (0 or 1).
    """
    if len(pop_proportions) != 2:
        raise ValueError("Currently only two populations are supported.")
    if abs(sum(pop_proportions) - 1.0) > 1e-6:
        raise ValueError("pop_proportions must sum to 1.0.")
    if n_cat_levels % 2 != 0:
        raise ValueError("n_cat_levels must be an even number.")
    if n_con_with_err > n_con_features:
        raise ValueError("n_con_with_err cannot exceed n_con_features.")
    if n_cat_with_err > n_cat_features:
        raise ValueError("n_cat_with_err cannot exceed n_cat_features.")

    rng = check_random_state(random_state)
    overlap_default = 0.01

    # 1. Sample population IDs (0 or 1)
    y = rng.choice(2, size=n_samples, p=pop_proportions)
    num_pop0 = int(np.sum(y == 0))
    num_pop1 = n_samples - num_pop0

    # 2. Continuous variables
    if n_con_features > 0:
        X_con = np.empty((n_samples, n_con_features), dtype=np.float64)
        for j in range(n_con_features):
            if j < n_con_with_err:
                # Feature with overlap/error
                mu0 = 0.0
                mu1 = float(2.0 * norm.ppf(1.0 - con_err_level / 2.0))
            else:
                # Feature without error (clear separation)
                mu0 = 0.0
                mu1 = float(2.0 * norm.ppf(1.0 - overlap_default / 2.0))

            X_con[y == 0, j] = rng.normal(loc=mu0, scale=1.0, size=num_pop0)
            X_con[y == 1, j] = rng.normal(loc=mu1, scale=1.0, size=num_pop1)
    else:
        X_con = np.empty((n_samples, 0), dtype=np.float64)

    # 3. Categorical variables
    if n_cat_features > 0:
        half_levels = n_cat_levels // 2
        X_cat_cols = []

        # (a) Error-free categorical features
        n_cat_no_err = n_cat_features - n_cat_with_err
        if n_cat_no_err > 0:
            right_p = (1.0 - overlap_default / 2.0) / half_levels
            wrong_p = (overlap_default / 2.0) / half_levels
            prob0 = np.concatenate([np.full(half_levels, right_p), np.full(half_levels, wrong_p)])
            prob1 = np.concatenate([np.full(half_levels, wrong_p), np.full(half_levels, right_p)])

            for _ in range(n_cat_no_err):
                col = np.empty(n_samples, dtype=np.int32)
                col[y == 0] = rng.choice(n_cat_levels, size=num_pop0, p=prob0)
                col[y == 1] = rng.choice(n_cat_levels, size=num_pop1, p=prob1)
                X_cat_cols.append(col)

        # (b) Categorical features with error
        if n_cat_with_err > 0:
            right_p_err = (1.0 - cat_err_level / 2.0) / half_levels
            wrong_p_err = (cat_err_level / 2.0) / half_levels
            prob0_err = np.concatenate([np.full(half_levels, right_p_err), np.full(half_levels, wrong_p_err)])
            prob1_err = np.concatenate([np.full(half_levels, wrong_p_err), np.full(half_levels, right_p_err)])

            for _ in range(n_cat_with_err):
                col = np.empty(n_samples, dtype=np.int32)
                col[y == 0] = rng.choice(n_cat_levels, size=num_pop0, p=prob0_err)
                col[y == 1] = rng.choice(n_cat_levels, size=num_pop1, p=prob1_err)
                X_cat_cols.append(col)

        X_cat = np.column_stack(X_cat_cols)
    else:
        X_cat = np.empty((n_samples, 0), dtype=np.int32)

    if return_dataframe:
        import pandas as pd

        data_dict = {}
        for j in range(n_con_features):
            data_dict[f"con_{j}"] = X_con[:, j]
        for j in range(n_cat_features):
            data_dict[f"cat_{j}"] = pd.Categorical(X_cat[:, j])
        df = pd.DataFrame(data_dict)
        return df, y

    return X_con, X_cat, y
