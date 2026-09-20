"""Input validation and feature extraction utilities for mixed-type data."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

from typing import Any, List, Optional, Tuple, Union
import numpy as np


def extract_mixed_features(
    X: Any,
    categorical_features: Optional[Union[str, List[Union[int, str]], np.ndarray]] = None,
    fit: bool = True,
    categories: Optional[List[np.ndarray]] = None,
    feature_names_in: Optional[np.ndarray] = None,
    con_indices: Optional[List[int]] = None,
    cat_indices: Optional[List[int]] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    List[np.ndarray],
    List[int],
    List[int],
    Optional[np.ndarray],
]:
    """Extract continuous and categorical matrices from mixed-type data.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
        Input dataset.
    categorical_features : 'auto', list of column names/indices, boolean mask, or None
        Specification of categorical features.
    fit : bool, default=True
        Whether this is called during fit (to learn categories) or predict (to validate).
    categories : list of ndarray, optional
        Pre-learned categories for each categorical column when fit=False.
    feature_names_in : ndarray of str, optional
        Pre-learned feature names when fit=False.
    con_indices : list of int, optional
        Indices of continuous columns when fit=False.
    cat_indices : list of int, optional
        Indices of categorical columns when fit=False.

    Returns
    -------
    X_con : ndarray of shape (n_samples, n_con) or None
        Continuous feature matrix.
    X_cat : ndarray of shape (n_samples, n_cat) with integer encoding (0-based) or None
        Categorical feature matrix.
    categories : list of ndarray
        Unique category levels for each categorical column.
    con_indices : list of int
        Column indices of continuous features.
    cat_indices : list of int
        Column indices of categorical features.
    feature_names_in : ndarray or None
        Feature names if available.
    """
    is_df = hasattr(X, "columns") and hasattr(X, "iloc")
    cols = list(X.columns) if is_df else None

    if fit:
        if is_df:
            feature_names = np.array(X.columns, dtype=object)
            n_features = len(X.columns)
            n_samples = len(X)
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got {X_arr.ndim}D array instead")
            n_samples, n_features = X_arr.shape
            feature_names = None

        if n_samples == 0:
            raise ValueError("Input data X contains 0 samples.")
        if n_features == 0:
            raise ValueError("Input data X contains 0 features.")

        # Determine categorical indices
        if categorical_features is None:
            cat_idx = []
            con_idx = list(range(n_features))
        elif isinstance(categorical_features, str) and categorical_features == "auto":
            if is_df:
                cat_idx = []
                for idx, col in enumerate(X.columns):
                    dtype = X[col].dtype
                    if (
                        str(dtype) in ("category", "object", "string")
                        or hasattr(dtype, "name")
                        and dtype.name in ("category", "object", "string")
                    ):
                        cat_idx.append(idx)
                con_idx = [i for i in range(n_features) if i not in cat_idx]
            else:
                cat_idx = []
                con_idx = list(range(n_features))
        elif isinstance(categorical_features, (list, tuple, np.ndarray)):
            cat_list = list(categorical_features)
            if len(cat_list) == n_features and all(isinstance(v, (bool, np.bool_)) for v in cat_list):
                cat_idx = [i for i, b in enumerate(cat_list) if b]
            elif all(isinstance(v, str) for v in cat_list):
                if not is_df:
                    raise ValueError(
                        "categorical_features specified by name, but X is not a DataFrame."
                    )
                col_to_idx = {col: i for i, col in enumerate(X.columns)}
                cat_idx = [col_to_idx[c] for c in cat_list]
            elif all(isinstance(v, (int, np.integer)) for v in cat_list):
                cat_idx = [int(v) for v in cat_list]
            else:
                raise ValueError("Invalid format for categorical_features.")
            con_idx = [i for i in range(n_features) if i not in cat_idx]
        else:
            raise ValueError(
                f"Unsupported categorical_features specification: {categorical_features}"
            )
    else:
        # Predict mode: re-use fitted indices
        con_idx = con_indices or []
        cat_idx = cat_indices or []
        feature_names = feature_names_in
        if is_df:
            n_features = len(X.columns)
            n_samples = len(X)
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got {X_arr.ndim}D array instead")
            n_samples, n_features = X_arr.shape

    if len(con_idx) == 0 and len(cat_idx) == 0:
        raise ValueError("At least one continuous or categorical variable must be present.")

    # Process Continuous Features
    X_con = None
    if len(con_idx) > 0:
        if is_df:
            con_data = X.iloc[:, con_idx].to_numpy(dtype=np.float64, copy=True)
        else:
            con_data = np.asarray(X_arr[:, con_idx], dtype=np.float64)
        if np.isnan(con_data).any() or np.isinf(con_data).any():
            raise ValueError("Continuous features contain NaN or Inf values.")
        X_con = con_data

    # Process Categorical Features
    X_cat = None
    learned_categories = list(categories) if categories is not None else []

    if len(cat_idx) > 0:
        encoded_cols = []
        for i, col_idx in enumerate(cat_idx):
            if is_df:
                raw_series = X.iloc[:, col_idx]
                vals = raw_series.to_numpy(copy=False)
            else:
                vals = X_arr[:, col_idx]

            # Convert to string representations or preserved types
            val_strs = np.array([str(v) if v is not None else "nan" for v in vals], dtype=object)

            if fit:
                uniq_vals = np.unique(val_strs)
                learned_categories.append(uniq_vals)
                cat_map = {v: idx for idx, v in enumerate(uniq_vals)}
            else:
                uniq_vals = learned_categories[i]
                cat_map = {v: idx for idx, v in enumerate(uniq_vals)}
                # Check for unseen categories
                unseen = set(val_strs) - set(uniq_vals)
                if len(unseen) > 0:
                    col_id = cols[col_idx] if cols else f"column {col_idx}"
                    formatted = ", ".join(f"'{u}'" for u in sorted(unseen))
                    raise ValueError(
                        f"Categorical variable '{col_id}' contains unseen levels: {formatted}"
                    )

            encoded = np.array([cat_map[v] for v in val_strs], dtype=np.int32)
            encoded_cols.append(encoded)

        X_cat = np.column_stack(encoded_cols)

    return X_con, X_cat, learned_categories, con_idx, cat_idx, feature_names
