"""Input validation and preprocessing routines for mixed continuous and categorical
data."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

import numpy as np
import scipy.sparse as sp


def _check_categorical_features(
    categorical_features, X, n_features=None, feature_names=None
):
    """Determine a boolean mask indicating which features are categorical.

    Follows scikit-learn's HistGradientBoosting conventions.

    Parameters
    ----------
    categorical_features : None, 'from_dtype', or array-like
        Specification of categorical features.
    X : array-like of shape (n_samples, n_features)
        The input data.
    n_features : int, optional
        Number of features in X.
    feature_names : array-like of str, optional
        Names of the features in X.

    Returns
    -------
    is_categorical : ndarray of shape (n_features,) of bool
        Boolean mask where True indicates a categorical feature.
    """
    if n_features is None:
        if hasattr(X, "shape"):
            n_features = X.shape[1]
        else:
            X_list = list(X)
            n_features = len(X_list[0]) if len(X_list) > 0 else 0

    if categorical_features is None:
        return np.zeros(n_features, dtype=bool)

    if isinstance(categorical_features, str):
        if categorical_features == "from_dtype":
            is_cat = np.zeros(n_features, dtype=bool)
            if hasattr(X, "dtypes"):
                for idx, dtype in enumerate(X.dtypes):
                    dtype_str = str(dtype)
                    if (
                        dtype_str == "category"
                        or dtype_str == "bool"
                        or hasattr(dtype, "categories")
                    ):
                        is_cat[idx] = True
            return is_cat
        else:
            raise ValueError(
                "categorical_features must be None, 'from_dtype', or array-like, "
                f"got {categorical_features!r}."
            )

    cat_arr = np.asarray(categorical_features)
    if cat_arr.dtype.kind == "b":
        if cat_arr.shape != (n_features,):
            raise ValueError(
                "categorical_features as a boolean mask must have shape "
                f"({n_features},), but got shape {cat_arr.shape}."
            )
        return cat_arr.copy()

    is_cat = np.zeros(n_features, dtype=bool)
    if cat_arr.dtype.kind in ("i", "u"):
        for idx in cat_arr:
            orig_idx = int(idx)
            pos_idx = orig_idx if orig_idx >= 0 else n_features + orig_idx
            if pos_idx < 0 or pos_idx >= n_features:
                raise ValueError(
                    f"Categorical feature index {orig_idx} is out of bounds for "
                    f"array with {n_features} features."
                )
            is_cat[pos_idx] = True
        return is_cat

    if cat_arr.dtype.kind in ("U", "O", "S"):
        if feature_names is None:
            raise ValueError(
                "Categorical features were passed as feature names, but no "
                "feature names are available for X."
            )
        feature_names_list = list(feature_names)
        for name in cat_arr:
            name_str = str(name)
            if name_str not in feature_names_list:
                raise ValueError(
                    f"Categorical feature name {name_str!r} not found in "
                    f"feature names: {feature_names_list}."
                )
            is_cat[feature_names_list.index(name_str)] = True
        return is_cat

    raise ValueError(
        "categorical_features must be None, 'from_dtype', or array-like of "
        f"bool, int, or str. Got {categorical_features!r} with dtype {cat_arr.dtype}."
    )


def _validate_and_split_data(
    X,
    categorical_features=None,
    feature_names=None,
    categories=None,
    con_weights=None,
    cat_weights=None,
    is_categorical=None,
):
    """Validate input data X and split into continuous and categorical arrays.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input dataset.
    categorical_features : specification of categorical features, optional
    feature_names : array-like of str, optional
    categories : list of arrays, optional
        Pre-existing category mappings (e.g. from fit).
    con_weights : array-like of shape (n_con,), optional
    cat_weights : array-like of shape (n_cat,), optional
    is_categorical : ndarray of shape (n_features,) of bool, optional

    Returns
    -------
    X_con : ndarray of shape (n_samples, n_con) of float64 or None
    X_cat : ndarray of shape (n_samples, n_cat) of int32 or None
    is_categorical : ndarray of shape (n_features,) of bool
    categories : list of ndarrays (unique categories per categorical feature)
    num_levels : ndarray of int32 or None
    con_wgts : ndarray of float64 or None
    cat_wgts : ndarray of float64 or None
    detected_feature_names : ndarray of str or None
    """
    detected_feature_names = None
    if hasattr(X, "columns"):
        detected_feature_names = np.asarray(X.columns, dtype=object)
    elif feature_names is not None:
        detected_feature_names = np.asarray(feature_names, dtype=object)

    if sp.issparse(X):
        raise TypeError(
            "A sparse matrix was passed, but dense data is required. "
            "Use X.toarray() to convert to a dense numpy array."
        )

    if hasattr(X, "ndim") and X.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got 1D array instead: shape={getattr(X, 'shape', None)}.\n"
            "Reshape your data either using array.reshape(-1, 1) if your data has a single feature "
            "or array.reshape(1, -1) if it contains a single sample."
        )

    if hasattr(X, "shape"):
        n_samples, n_features = X.shape[0], X.shape[1]
    else:
        X_list = list(X)
        n_samples = len(X_list)
        if n_samples == 0:
            raise ValueError(
                f"Found array with 0 sample(s) (shape=(0, 0)) while a minimum of 1 is required."
            )
        n_features = len(X_list[0]) if hasattr(X_list[0], "__len__") else 0

    if n_samples == 0:
        raise ValueError(
            f"Found array with 0 sample(s) (shape=({n_samples}, {n_features})) while a minimum of 1 is required."
        )
    if n_features == 0:
        raise ValueError(
            f"Found array with 0 feature(s) (shape=({n_samples}, 0)) while a minimum of 1 is required."
        )

    if np.iscomplexobj(X):
        raise ValueError("Complex data not supported")

    if is_categorical is None:
        is_categorical = _check_categorical_features(
            categorical_features,
            X,
            n_features=n_features,
            feature_names=detected_feature_names,
        )
    elif len(is_categorical) != n_features:
        raise ValueError(
            f"X has {n_features} features, but KamilaClustering is expecting "
            f"{len(is_categorical)} features as input."
        )

    con_indices = np.where(~is_categorical)[0]
    cat_indices = np.where(is_categorical)[0]
    n_con = len(con_indices)
    n_cat = len(cat_indices)

    # Continuous features
    if n_con > 0:
        if hasattr(X, "iloc"):
            con_raw = X.iloc[:, con_indices].to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(X)
            con_raw = arr[:, con_indices].astype(np.float64)
        if not np.all(np.isfinite(con_raw)):
            raise ValueError("Continuous features contain NaN or infinite values.")
        X_con = np.ascontiguousarray(con_raw, dtype=np.float64)

        if con_weights is not None:
            con_wgts = np.ascontiguousarray(con_weights, dtype=np.float64)
            if con_wgts.shape != (n_con,):
                raise ValueError(
                    f"con_weights shape {con_wgts.shape} does not match number of "
                    f"continuous features ({n_con})."
                )
            if np.any(con_wgts <= 0):
                raise ValueError("All con_weights must be strictly positive.")
        else:
            con_wgts = np.ones(n_con, dtype=np.float64)
    else:
        X_con = None
        con_wgts = None

    # Categorical features
    if n_cat > 0:
        encoded_cols = []
        new_categories = []
        num_levels_list = []

        for idx_pos, c_idx in enumerate(cat_indices):
            if hasattr(X, "iloc"):
                col = X.iloc[:, c_idx].to_numpy()
            else:
                arr = np.asarray(X)
                col = arr[:, c_idx]

            if categories is not None and idx_pos < len(categories):
                cats = categories[idx_pos]
            else:
                cats = np.unique(col)

            new_categories.append(cats)
            num_levels_list.append(len(cats))

            cat_map = {val: i for i, val in enumerate(cats)}
            encoded_col = np.empty(n_samples, dtype=np.int32)
            for i, val in enumerate(col):
                if val not in cat_map:
                    raise ValueError(
                        f"Unseen category {val!r} encountered in categorical "
                        f"feature {c_idx}."
                    )
                encoded_col[i] = cat_map[val]
            encoded_cols.append(encoded_col)

        X_cat = np.ascontiguousarray(np.column_stack(encoded_cols), dtype=np.int32)
        categories = new_categories
        num_levels = np.ascontiguousarray(num_levels_list, dtype=np.int32)

        if cat_weights is not None:
            cat_wgts = np.ascontiguousarray(cat_weights, dtype=np.float64)
            if cat_wgts.shape != (n_cat,):
                raise ValueError(
                    f"cat_weights shape {cat_wgts.shape} does not match number of "
                    f"categorical features ({n_cat})."
                )
            if np.any(cat_wgts <= 0):
                raise ValueError("All cat_weights must be strictly positive.")
        else:
            cat_wgts = np.ones(n_cat, dtype=np.float64)
    else:
        X_cat = None
        categories = []
        num_levels = None
        cat_wgts = None

    return (
        X_con,
        X_cat,
        is_categorical,
        categories,
        num_levels,
        con_wgts,
        cat_wgts,
        detected_feature_names,
    )
