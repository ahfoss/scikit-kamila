"""Tests for KamilaClustering estimator and R reference data parity."""

# Authors: Alexander Foss <alexanderhfoss@gmail.com>
#          Marianthi Markatou <markatou@buffalo.edu>
# License: BSD 3 clause

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from kamila import KamilaClustering, _kamila_cpp
from kamila._validation import _check_categorical_features, _validate_and_split_data

DATA_DIR = Path(__file__).parent / "data"


def load_fixture(name):
    """Load a reference JSON fixture."""
    fixture_file = DATA_DIR / name
    with open(fixture_file, "r", encoding="utf-8") as f:
        return json.load(f)


def test_kamila_version():
    """Verify package exposes __version__ string and C++ version."""
    import kamila

    assert hasattr(kamila, "__version__")
    assert isinstance(kamila.__version__, str)
    assert _kamila_cpp.get_cpp_version() >= 1


def test_kamila_initialization():
    """Verify KamilaClustering can be instantiated with default and custom params."""
    kam = KamilaClustering()
    assert kam.n_clusters == 2
    assert kam.n_init == 10
    assert kam.max_iter == 25
    assert kam.cat_bandwidth == 0.025
    assert kam.random_state is None
    assert kam.categorical_features is None
    assert kam.con_weights is None
    assert kam.cat_weights is None


# =============================================================================
# Exact Parity Tests with R 'kamila' Reference Data
# =============================================================================


def test_parity_small_mixed():
    """Verify exact 1-iteration and converged parity on small mixed dataset."""
    data = load_fixture("reference_small_mixed.json")
    con_data = np.asarray(data["con_data"], dtype=np.float64)
    cat_data = np.asarray(data["cat_data"], dtype=np.int32)
    X = np.hstack([con_data, cat_data])

    init_means = np.asarray(data["init_means"], dtype=np.float64)
    init_log_probs = [np.asarray(lp, dtype=np.float64) for lp in data["init_log_probs"]]

    # 1 iteration
    kam_1 = KamilaClustering(
        n_clusters=data["num_clust"],
        categorical_features=[2, 3],
        max_iter=1,
        cat_bandwidth=data["cat_bw"],
        con_weights=data["con_weights"],
        cat_weights=data["cat_weights"],
        init_means=init_means,
        init_log_probs=init_log_probs,
    )
    kam_1.fit(X)

    expected_1 = data["one_iteration"]
    np.testing.assert_array_equal(kam_1.labels_, expected_1["final_membership"])
    np.testing.assert_allclose(
        kam_1.cluster_centers_con_, expected_1["final_means"], rtol=1e-7, atol=1e-7
    )
    for j in range(len(init_log_probs)):
        np.testing.assert_allclose(
            kam_1.cluster_centers_cat_[j],
            expected_1["final_log_probs"][j],
            rtol=1e-7,
            atol=1e-7,
        )
    assert kam_1.total_log_lik_ == pytest.approx(expected_1["total_log_lik"], rel=1e-7)
    assert kam_1.cat_log_lik_ == pytest.approx(expected_1["cat_log_lik"], rel=1e-7)
    assert kam_1.win_dist_ == pytest.approx(expected_1["win_dist"], rel=1e-6)
    assert kam_1.n_iter_ == expected_1["num_iter"]

    # Converged
    kam_conv = KamilaClustering(
        n_clusters=data["num_clust"],
        categorical_features=[2, 3],
        max_iter=20,
        cat_bandwidth=data["cat_bw"],
        con_weights=data["con_weights"],
        cat_weights=data["cat_weights"],
        init_means=init_means,
        init_log_probs=init_log_probs,
    )
    kam_conv.fit(X)

    expected_conv = data["converged"]
    np.testing.assert_array_equal(kam_conv.labels_, expected_conv["final_membership"])
    np.testing.assert_allclose(
        kam_conv.cluster_centers_con_,
        expected_conv["final_means"],
        rtol=1e-7,
        atol=1e-7,
    )
    for j in range(len(init_log_probs)):
        np.testing.assert_allclose(
            kam_conv.cluster_centers_cat_[j],
            expected_conv["final_log_probs"][j],
            rtol=1e-7,
            atol=1e-7,
        )
    assert kam_conv.total_log_lik_ == pytest.approx(
        expected_conv["total_log_lik"], rel=1e-7
    )
    assert kam_conv.cat_log_lik_ == pytest.approx(
        expected_conv["cat_log_lik"], rel=1e-7
    )
    assert kam_conv.win_dist_ == pytest.approx(expected_conv["win_dist"], rel=1e-6)
    assert kam_conv.n_iter_ == expected_conv["num_iter"]

    # Predict
    preds = kam_conv.predict(X)
    np.testing.assert_array_equal(preds, kam_conv.labels_)


def test_parity_medium_mixed():
    """Verify exact 1-iteration and converged parity on medium mixed dataset."""
    data = load_fixture("reference_medium_mixed.json")
    con_data = np.asarray(data["con_data"], dtype=np.float64)
    cat_data = np.asarray(data["cat_data"], dtype=np.int32)
    X = np.hstack([con_data, cat_data])

    init_means = np.asarray(data["init_means"], dtype=np.float64)
    init_log_probs = [np.asarray(lp, dtype=np.float64) for lp in data["init_log_probs"]]

    # 1 iteration
    kam_1 = KamilaClustering(
        n_clusters=data["num_clust"],
        categorical_features=[3, 4],
        max_iter=1,
        cat_bandwidth=data["cat_bw"],
        con_weights=data["con_weights"],
        cat_weights=data["cat_weights"],
        init_means=init_means,
        init_log_probs=init_log_probs,
    )
    kam_1.fit(X)

    expected_1 = data["one_iteration"]
    np.testing.assert_array_equal(kam_1.labels_, expected_1["final_membership"])
    np.testing.assert_allclose(
        kam_1.cluster_centers_con_, expected_1["final_means"], rtol=1e-7, atol=1e-7
    )
    for j in range(len(init_log_probs)):
        np.testing.assert_allclose(
            kam_1.cluster_centers_cat_[j],
            expected_1["final_log_probs"][j],
            rtol=1e-7,
            atol=1e-7,
        )
    assert kam_1.total_log_lik_ == pytest.approx(expected_1["total_log_lik"], rel=1e-7)
    assert kam_1.cat_log_lik_ == pytest.approx(expected_1["cat_log_lik"], rel=1e-7)
    assert kam_1.win_dist_ == pytest.approx(expected_1["win_dist"], rel=1e-6)
    assert kam_1.n_iter_ == expected_1["num_iter"]

    # Converged
    kam_conv = KamilaClustering(
        n_clusters=data["num_clust"],
        categorical_features=[3, 4],
        max_iter=25,
        cat_bandwidth=data["cat_bw"],
        con_weights=data["con_weights"],
        cat_weights=data["cat_weights"],
        init_means=init_means,
        init_log_probs=init_log_probs,
    )
    kam_conv.fit(X)

    expected_conv = data["converged"]
    np.testing.assert_array_equal(kam_conv.labels_, expected_conv["final_membership"])
    np.testing.assert_allclose(
        kam_conv.cluster_centers_con_,
        expected_conv["final_means"],
        rtol=1e-7,
        atol=1e-7,
    )
    for j in range(len(init_log_probs)):
        np.testing.assert_allclose(
            kam_conv.cluster_centers_cat_[j],
            expected_conv["final_log_probs"][j],
            rtol=1e-7,
            atol=1e-7,
        )
    assert kam_conv.total_log_lik_ == pytest.approx(
        expected_conv["total_log_lik"], rel=1e-7
    )
    assert kam_conv.cat_log_lik_ == pytest.approx(
        expected_conv["cat_log_lik"], rel=1e-7
    )
    assert kam_conv.win_dist_ == pytest.approx(expected_conv["win_dist"], rel=1e-6)
    assert kam_conv.n_iter_ == expected_conv["num_iter"]


def test_parity_continuous_only():
    """Verify exact numerical parity on continuous-only dataset."""
    data = load_fixture("reference_continuous_only.json")
    con_data = np.asarray(data["con_data"], dtype=np.float64)
    init_means = np.asarray(data["init_means"], dtype=np.float64)

    # 1 iteration
    kam_1 = KamilaClustering(
        n_clusters=data["num_clust"],
        categorical_features=None,
        max_iter=1,
        con_weights=data["con_weights"],
        init_means=init_means,
    )
    kam_1.fit(con_data)

    expected_1 = data["one_iteration"]
    np.testing.assert_array_equal(kam_1.labels_, expected_1["final_membership"])
    np.testing.assert_allclose(
        kam_1.cluster_centers_con_, expected_1["final_means"], rtol=1e-7, atol=1e-7
    )
    assert kam_1.total_log_lik_ == pytest.approx(expected_1["total_log_lik"], rel=1e-7)
    assert kam_1.win_dist_ == pytest.approx(expected_1["win_dist"], rel=1e-6)
    assert kam_1.n_iter_ == expected_1["num_iter"]
    assert kam_1.cluster_centers_cat_ is None
    assert kam_1.cat_log_lik_ is None

    # Converged
    kam_conv = KamilaClustering(
        n_clusters=data["num_clust"],
        categorical_features=None,
        max_iter=25,
        con_weights=data["con_weights"],
        init_means=init_means,
    )
    kam_conv.fit(con_data)

    expected_conv = data["converged"]
    np.testing.assert_array_equal(kam_conv.labels_, expected_conv["final_membership"])
    np.testing.assert_allclose(
        kam_conv.cluster_centers_con_,
        expected_conv["final_means"],
        rtol=1e-7,
        atol=1e-7,
    )
    assert kam_conv.total_log_lik_ == pytest.approx(
        expected_conv["total_log_lik"], rel=1e-7
    )
    assert kam_conv.win_dist_ == pytest.approx(expected_conv["win_dist"], rel=1e-6)
    assert kam_conv.n_iter_ == expected_conv["num_iter"]

    preds = kam_conv.predict(con_data)
    np.testing.assert_array_equal(preds, kam_conv.labels_)


def test_parity_categorical_only():
    """Verify exact numerical parity on categorical-only dataset."""
    data = load_fixture("reference_categorical_only.json")
    cat_data = np.asarray(data["cat_data"], dtype=np.int32)
    init_log_probs = [np.asarray(lp, dtype=np.float64) for lp in data["init_log_probs"]]

    # 1 iteration
    kam_1 = KamilaClustering(
        n_clusters=data["num_clust"],
        categorical_features=[True, True],
        max_iter=1,
        cat_bandwidth=data["cat_bw"],
        cat_weights=data["cat_weights"],
        init_log_probs=init_log_probs,
    )
    kam_1.fit(cat_data)

    expected_1 = data["one_iteration"]
    np.testing.assert_array_equal(kam_1.labels_, expected_1["final_membership"])
    for j in range(len(init_log_probs)):
        np.testing.assert_allclose(
            kam_1.cluster_centers_cat_[j],
            expected_1["final_log_probs"][j],
            rtol=1e-7,
            atol=1e-7,
        )
    assert kam_1.cat_log_lik_ == pytest.approx(expected_1["cat_log_lik"], rel=1e-7)
    assert kam_1.n_iter_ == expected_1["num_iter"]
    assert kam_1.cluster_centers_con_ is None
    assert kam_1.total_log_lik_ is None
    assert kam_1.win_dist_ is None

    # Converged
    kam_conv = KamilaClustering(
        n_clusters=data["num_clust"],
        categorical_features=[True, True],
        max_iter=25,
        cat_bandwidth=data["cat_bw"],
        cat_weights=data["cat_weights"],
        init_log_probs=init_log_probs,
    )
    kam_conv.fit(cat_data)

    expected_conv = data["converged"]
    np.testing.assert_array_equal(kam_conv.labels_, expected_conv["final_membership"])
    for j in range(len(init_log_probs)):
        np.testing.assert_allclose(
            kam_conv.cluster_centers_cat_[j],
            expected_conv["final_log_probs"][j],
            rtol=1e-7,
            atol=1e-7,
        )
    assert kam_conv.cat_log_lik_ == pytest.approx(
        expected_conv["cat_log_lik"], rel=1e-7
    )
    assert kam_conv.n_iter_ == expected_conv["num_iter"]

    preds = kam_conv.predict(cat_data)
    np.testing.assert_array_equal(preds, kam_conv.labels_)


# =============================================================================
# Additional Functionality and Edge Cases
# =============================================================================


def test_kamila_random_init_and_reproducibility():
    """Verify multi-start random init and random_state reproducibility."""
    rng = np.random.RandomState(42)
    X_con = rng.randn(40, 2)
    X_cat = rng.choice(["low", "med", "high"], size=(40, 2))
    X = np.hstack([X_con, X_cat])

    kam1 = KamilaClustering(
        n_clusters=2,
        categorical_features=[2, 3],
        n_init=3,
        max_iter=10,
        random_state=123,
    )
    labels1 = kam1.fit_predict(X)

    kam2 = KamilaClustering(
        n_clusters=2,
        categorical_features=[2, 3],
        n_init=3,
        max_iter=10,
        random_state=123,
    )
    labels2 = kam2.fit_predict(X)

    np.testing.assert_array_equal(labels1, labels2)
    assert kam1.inertia_ == pytest.approx(kam2.inertia_)


def test_kamila_random_init_single_modalities():
    """Verify random initialization on continuous-only and categorical-only data."""
    # Continuous-only random init
    X_con = np.array([[1.0, 2.0], [1.2, 2.1], [5.0, 5.0], [5.2, 5.1]])
    kam_con = KamilaClustering(n_clusters=2, n_init=2, random_state=42)
    kam_con.fit(X_con)
    assert kam_con.labels_.shape == (4,)
    assert kam_con.cluster_centers_cat_ is None

    # Categorical-only random init
    X_cat = np.array([["a", "x"], ["a", "x"], ["b", "y"], ["b", "y"]])
    kam_cat = KamilaClustering(
        n_clusters=2, categorical_features=[0, 1], n_init=2, random_state=42
    )
    kam_cat.fit(X_cat)
    assert kam_cat.labels_.shape == (4,)
    assert kam_con.cluster_centers_con_ is not None


def test_categorical_features_specifications():
    """Verify various ways to specify categorical features."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "num2": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
            "cat1": pd.Series(["a", "b", "a", "b", "a", "b"], dtype="category"),
            "bool1": [True, False, True, False, True, False],
        }
    )

    # 1. from_dtype
    kam = KamilaClustering(
        n_clusters=2, categorical_features="from_dtype", random_state=42
    )
    kam.fit(df)
    np.testing.assert_array_equal(kam.is_categorical_, [False, False, True, True])
    assert hasattr(kam, "feature_names_in_")

    # 2. list of column names
    kam = KamilaClustering(
        n_clusters=2, categorical_features=["num2", "cat1"], random_state=42
    )
    kam.fit(df)
    np.testing.assert_array_equal(kam.is_categorical_, [False, True, True, False])

    # 3. negative indices
    kam = KamilaClustering(n_clusters=2, categorical_features=[-1, 2], random_state=42)
    kam.fit(df)
    np.testing.assert_array_equal(kam.is_categorical_, [False, False, True, True])

    # 4. boolean mask
    kam = KamilaClustering(
        n_clusters=2,
        categorical_features=[True, False, True, False],
        random_state=42,
    )
    kam.fit(df)
    np.testing.assert_array_equal(kam.is_categorical_, [True, False, True, False])


def test_categorical_from_dtype_variants():
    """Verify from_dtype on DataFrame without category columns and numpy array."""
    import pandas as pd

    df_num = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    mask = _check_categorical_features("from_dtype", df_num)
    np.testing.assert_array_equal(mask, [False, False])

    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask_arr = _check_categorical_features("from_dtype", arr)
    np.testing.assert_array_equal(mask_arr, [False, False])


def test_categories_sorting_and_uncomparable_types():
    """Verify categorical feature handling with mixed category types."""
    X_mixed = np.array([[1, "a"], [2, "b"], [1, "b"], [2, "a"]], dtype=object)
    kam = KamilaClustering(n_clusters=2, categorical_features=[0, 1], random_state=42)
    kam.fit(X_mixed)
    assert kam.labels_.shape == (4,)

    # Feature names passed explicitly
    _, _, _, _, _, _, _, names = _validate_and_split_data(
        np.array([[1.0, 2.0], [3.0, 4.0]]), feature_names=["col_a", "col_b"]
    )
    np.testing.assert_array_equal(names, ["col_a", "col_b"])

    # _check_categorical_features called without n_features
    mask_list = _check_categorical_features(None, [[1.0, 2.0]])
    np.testing.assert_array_equal(mask_list, [False, False])

    mask_empty = _check_categorical_features(None, [])
    assert len(mask_empty) == 0


def test_validation_errors():
    """Test input validation exceptions and error messages."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Invalid n_clusters
    with pytest.raises(ValueError, match="n_clusters must be an integer >= 1"):
        KamilaClustering(n_clusters=0).fit(X)

    # n_samples < n_clusters
    with pytest.raises(ValueError, match="n_samples=2 should be >= n_clusters=5"):
        KamilaClustering(n_clusters=5).fit(X)

    # Invalid n_init
    with pytest.raises(ValueError, match="n_init must be an integer >= 1"):
        KamilaClustering(n_init=0).fit(X)

    # Invalid max_iter
    with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
        KamilaClustering(max_iter=0).fit(X)

    # Invalid cat_bandwidth
    with pytest.raises(
        ValueError, match="cat_bandwidth must be a strictly positive number"
    ):
        KamilaClustering(cat_bandwidth=0).fit(X)

    # Invalid categorical_features string
    with pytest.raises(ValueError, match="categorical_features must be None"):
        KamilaClustering(categorical_features="invalid_string").fit(X)

    # Boolean mask shape mismatch
    with pytest.raises(
        ValueError, match="categorical_features as a boolean mask must have shape"
    ):
        KamilaClustering(categorical_features=[True, False, True]).fit(X)

    # Index out of bounds
    with pytest.raises(ValueError, match="is out of bounds"):
        KamilaClustering(categorical_features=[10]).fit(X)

    # Feature name not in list
    with pytest.raises(
        ValueError, match="Categorical features were passed as feature names"
    ):
        _check_categorical_features(["feat1"], X, n_features=2, feature_names=None)

    with pytest.raises(ValueError, match="not found in feature names"):
        _check_categorical_features(
            ["non_existent"], X, n_features=2, feature_names=["col1", "col2"]
        )

    # Invalid categorical_features type
    with pytest.raises(ValueError, match="categorical_features must be None"):
        _check_categorical_features([1.5, 2.5], X, n_features=2)

    # 1D array
    with pytest.raises(ValueError, match="Expected 2D array"):
        _validate_and_split_data(np.array([1.0, 2.0]))

    # Empty data
    with pytest.raises(ValueError, match="Empty data passed"):
        _validate_and_split_data([])

    # Zero features
    with pytest.raises(ValueError, match="X must have at least one feature"):
        _validate_and_split_data(np.empty((5, 0)))

    # NaN / Inf in continuous
    with pytest.raises(ValueError, match="Continuous features contain NaN or infinite"):
        _validate_and_split_data(np.array([[np.nan, 1.0], [2.0, 3.0]]))

    # Invalid continuous weights shape / values
    with pytest.raises(ValueError, match="con_weights shape"):
        _validate_and_split_data(np.array([[1.0, 2.0], [3.0, 4.0]]), con_weights=[1.0])

    with pytest.raises(ValueError, match="All con_weights must be strictly positive"):
        _validate_and_split_data(
            np.array([[1.0, 2.0], [3.0, 4.0]]), con_weights=[1.0, -0.5]
        )

    # Invalid categorical weights shape / values
    with pytest.raises(ValueError, match="cat_weights shape"):
        _validate_and_split_data(
            np.array([["a", "b"], ["c", "d"]]),
            categorical_features=[True, True],
            cat_weights=[1.0],
        )

    with pytest.raises(ValueError, match="All cat_weights must be strictly positive"):
        _validate_and_split_data(
            np.array([["a", "b"], ["c", "d"]]),
            categorical_features=[True, True],
            cat_weights=[1.0, 0.0],
        )

    # Predict before fit
    kam = KamilaClustering()
    with pytest.raises(NotFittedError):
        kam.predict(X)

    # Unseen category in predict
    kam_unseen = KamilaClustering(
        n_clusters=2, categorical_features=[0], random_state=42
    )
    kam_unseen.fit([["red"], ["blue"]])
    with pytest.raises(ValueError, match="Unseen category .*green.* encountered"):
        kam_unseen.predict([["green"]])

    # Mismatched number of features in predict
    kam_mismatch = KamilaClustering(n_clusters=2, random_state=42)
    kam_mismatch.fit(X)
    with pytest.raises(ValueError, match="features, but KamilaClustering is expecting"):
        kam_mismatch.predict(np.array([[1.0]]))
