"""Regression tests for bugs reported in GitHub issues #16, #17, #19, #20 and #22."""

import numpy as np
import pytest

from kamila import KamilaClustering, _kamila_cpp


def _two_blobs(seed=0):
    """Two well-separated 2D Gaussian blobs of 100 points each."""
    rng = np.random.default_rng(seed)
    return np.vstack(
        [rng.normal([0, 0], 0.5, (100, 2)), rng.normal([6, 6], 0.5, (100, 2))]
    )


def _mixed_data(seed=0):
    """Two blobs plus one categorical column with 3 levels."""
    rng = np.random.default_rng(seed)
    return np.column_stack([_two_blobs(seed), rng.integers(0, 3, 200)])


def _uniform_log_probs(n_clusters, n_levels):
    return np.log(np.full((n_clusters, n_levels), 1.0 / n_levels))


# =============================================================================
# #16: Empty-cluster (degenerate) fits
# =============================================================================


def test_empty_cluster_is_reinitialized_by_membership():
    """An empty cluster is refilled from a random sample instead of dropped."""
    X = _two_blobs()
    init_means = np.array([[0.0, 0.0], [6.0, 6.0], [100.0, 100.0]])

    km = KamilaClustering(3, init_means=init_means, random_state=0).fit(X)
    km_again = KamilaClustering(3, init_means=init_means, random_state=0).fit(X)

    np.testing.assert_array_equal(np.unique(km.labels_), [0, 1, 2])
    assert np.all(km.cluster_centers_con_ >= X.min(axis=0))
    assert np.all(km.cluster_centers_con_ <= X.max(axis=0))
    np.testing.assert_array_equal(km.labels_, km_again.labels_)
    np.testing.assert_array_equal(
        km.cluster_centers_con_, km_again.cluster_centers_con_
    )


def _far_init_loop(seed, max_iter):
    """Direct C++ call where cluster 2 starts far from all data."""
    X = _two_blobs()
    return X, _kamila_cpp.kamila_loop_cpp(
        con_data=X,
        cat_data=None,
        n_samples=len(X),
        n_con=2,
        n_cat=0,
        n_clusters=3,
        con_weights=np.ones(2),
        cat_weights=None,
        num_levels=None,
        init_means=np.array([[0.0, 0.0], [6.0, 6.0], [100.0, 100.0]]),
        init_log_probs=None,
        cat_bw=0.025,
        max_iter=max_iter,
        has_con=True,
        has_cat=False,
        seed=seed,
    )


def test_cpp_empty_cluster_takes_one_random_member():
    """After one iteration the refilled cluster holds exactly one moved sample,
    and its mean is that sample."""
    X, res = _far_init_loop(seed=1, max_iter=1)

    assert res["degenerate_soln"] is False
    membership = np.asarray(res["final_membership"])
    (moved,) = np.flatnonzero(membership == 2)
    final_means = np.asarray(res["final_means"]).reshape(3, 2)
    np.testing.assert_allclose(final_means[2], X[moved])


def test_cpp_empty_cluster_reinitialization_is_seeded():
    _, a = _far_init_loop(seed=7, max_iter=25)
    _, b = _far_init_loop(seed=7, max_iter=25)

    assert a["degenerate_soln"] is False
    assert set(a["final_membership"]) == {0, 1, 2}
    assert a["final_membership"] == b["final_membership"]
    assert a["final_means"] == b["final_means"]


def test_cpp_empty_cluster_keeps_previous_mean():
    """With fewer samples than clusters no sample can be moved; the empty
    cluster keeps its previous mean instead of being reset to zeros."""
    con_data = np.array([[0.0, 0.0], [0.1, 0.1]], dtype=np.float64)
    init_means = np.array([[0.0, 0.0], [0.1, 0.1], [1000.0, 1000.0]], dtype=np.float64)

    res = _kamila_cpp.kamila_loop_cpp(
        con_data=con_data,
        cat_data=None,
        n_samples=2,
        n_con=2,
        n_cat=0,
        n_clusters=3,
        con_weights=np.array([1.0, 1.0], dtype=np.float64),
        cat_weights=None,
        num_levels=None,
        init_means=init_means,
        init_log_probs=None,
        cat_bw=0.025,
        max_iter=5,
        has_con=True,
        has_cat=False,
    )

    assert res["degenerate_soln"] is True
    final_means = np.asarray(res["final_means"]).reshape(3, 2)
    np.testing.assert_allclose(final_means[2], [1000.0, 1000.0])


def test_categorical_only_empty_clusters_are_reinitialized():
    """5 clusters on one 4-level feature: argmax alone can fill at most 4, so
    the fifth is filled by moving a random sample into it."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, (300, 1))

    km = KamilaClustering(5, categorical_features=[0], n_init=10, random_state=0).fit(X)

    np.testing.assert_array_equal(np.unique(km.labels_), [0, 1, 2, 3, 4])
    assert np.isfinite(km.inertia_)


# =============================================================================
# #17: Validation of explicit initializations
# =============================================================================


@pytest.mark.parametrize(
    "init_means",
    [
        np.zeros((1, 2)),  # too few clusters
        np.zeros((3, 3)),  # too many continuous features
        np.zeros(6),  # right size, wrong dimensionality
    ],
)
def test_init_means_wrong_shape_raises(init_means):
    with pytest.raises(ValueError, match="init_means"):
        KamilaClustering(3, init_means=init_means).fit(_two_blobs())


def test_init_means_non_finite_raises():
    init_means = np.array([[0.0, 0.0], [np.nan, 6.0]])
    with pytest.raises(ValueError, match="init_means"):
        KamilaClustering(2, init_means=init_means).fit(_two_blobs())


@pytest.mark.parametrize(
    "init_log_probs",
    [
        [_uniform_log_probs(2, 3), _uniform_log_probs(2, 3)],  # too many features
        [_uniform_log_probs(2, 2)],  # too few levels
        [_uniform_log_probs(3, 3)],  # too many clusters
        [np.array([[0.5, -1.0, -1.0], [-1.0, -1.0, -1.0]])],  # positive log-prob
        [np.array([[np.nan, -1.0, -1.0], [-1.0, -1.0, -1.0]])],  # NaN
    ],
)
def test_init_log_probs_invalid_raises(init_log_probs):
    X = np.random.default_rng(0).integers(0, 3, (50, 1))
    with pytest.raises(ValueError, match="init_log_probs"):
        KamilaClustering(
            2, categorical_features=[0], init_log_probs=init_log_probs
        ).fit(X)


def test_partial_init_on_mixed_data_raises():
    X = _mixed_data()
    with pytest.raises(ValueError, match="init_means"):
        KamilaClustering(
            2, categorical_features=[2], init_log_probs=[_uniform_log_probs(2, 3)]
        ).fit(X)
    with pytest.raises(ValueError, match="init_log_probs"):
        KamilaClustering(
            2, categorical_features=[2], init_means=np.array([[0.0, 0.0], [6, 6]])
        ).fit(X)


def test_init_for_absent_feature_type_raises():
    X_cat = np.random.default_rng(0).integers(0, 3, (50, 1))
    with pytest.raises(ValueError, match="init_means"):
        KamilaClustering(
            2,
            categorical_features=[0],
            init_means=np.zeros((2, 1)),
            init_log_probs=[_uniform_log_probs(2, 3)],
        ).fit(X_cat)

    with pytest.raises(ValueError, match="init_log_probs"):
        KamilaClustering(
            2,
            init_means=np.array([[0.0, 0.0], [6.0, 6.0]]),
            init_log_probs=[_uniform_log_probs(2, 3)],
        ).fit(_two_blobs())


def _cpp_loop_kwargs(**overrides):
    """Valid arguments for a direct mixed-data kamila_loop_cpp call."""
    X = _mixed_data()
    kwargs = dict(
        con_data=np.ascontiguousarray(X[:, :2]),
        cat_data=np.ascontiguousarray(X[:, 2:], dtype=np.int32),
        n_samples=200,
        n_con=2,
        n_cat=1,
        n_clusters=2,
        con_weights=np.ones(2),
        cat_weights=np.ones(1),
        num_levels=np.array([3], dtype=np.int32),
        init_means=np.array([[0.0, 0.0], [6.0, 6.0]]),
        init_log_probs=[_uniform_log_probs(2, 3)],
        cat_bw=0.025,
        max_iter=10,
        has_con=True,
        has_cat=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_cpp_bindings_valid_call_succeeds():
    res = _kamila_cpp.kamila_loop_cpp(**_cpp_loop_kwargs())
    assert len(res["final_membership"]) == 200


@pytest.mark.parametrize(
    "overrides",
    [
        {"init_means": np.zeros((1, 2))},
        {"init_log_probs": [_uniform_log_probs(2, 2)]},
        {"init_log_probs": [_uniform_log_probs(2, 3), _uniform_log_probs(2, 3)]},
    ],
)
def test_cpp_bindings_reject_wrong_init_sizes(overrides):
    """The bindings must not hand undersized buffers to the C++ core."""
    with pytest.raises(ValueError, match="init_"):
        _kamila_cpp.kamila_loop_cpp(**_cpp_loop_kwargs(**overrides))


# =============================================================================
# #19: cat_bandwidth upper bound
# =============================================================================


def test_cat_bandwidth_above_one_raises():
    X = np.random.default_rng(0).integers(0, 3, (200, 1))
    with pytest.raises(ValueError, match="cat_bandwidth"):
        KamilaClustering(2, categorical_features=[0], cat_bandwidth=1.5).fit(X)


@pytest.mark.parametrize("cat_bandwidth", [0.0, 0.025, 0.5, 1.0])
def test_categorical_probabilities_are_valid(cat_bandwidth):
    """Every accepted bandwidth yields proper probability distributions."""
    km = KamilaClustering(
        2, categorical_features=[2], cat_bandwidth=cat_bandwidth, random_state=0
    ).fit(_mixed_data())

    for log_probs in km.cluster_centers_cat_:
        probs = np.exp(log_probs)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)


# =============================================================================
# #20: Feature-name validation in predict
# =============================================================================


@pytest.fixture
def fitted_on_dataframe():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": np.r_[rng.normal(0, 1, 100), rng.normal(10, 1, 100)],
            "b": rng.normal(0, 1, 200),
        }
    )
    return KamilaClustering(2, random_state=0).fit(df), df


def test_predict_rejects_reordered_columns(fitted_on_dataframe):
    km, df = fitted_on_dataframe
    with pytest.raises(ValueError, match="feature names should match"):
        km.predict(df[["b", "a"]])


def test_predict_rejects_renamed_columns(fitted_on_dataframe):
    km, df = fitted_on_dataframe
    with pytest.raises(ValueError, match="feature names should match"):
        km.predict(df.rename(columns={"b": "c"}))


def test_predict_warns_when_names_missing(fitted_on_dataframe):
    km, df = fitted_on_dataframe
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        km.predict(df.to_numpy())


def test_predict_warns_when_fitted_without_names(fitted_on_dataframe):
    _, df = fitted_on_dataframe
    km = KamilaClustering(2, random_state=0).fit(df.to_numpy())
    with pytest.warns(UserWarning, match="fitted without feature names"):
        km.predict(df)


def test_refit_without_names_clears_feature_names_in(fitted_on_dataframe):
    km, df = fitted_on_dataframe
    assert hasattr(km, "feature_names_in_")
    km.fit(df.to_numpy())
    assert not hasattr(km, "feature_names_in_")


# =============================================================================
# #22: KDE quartiles for n = 1 (mod 4)
# =============================================================================


@pytest.mark.parametrize("n_samples", [5, 9, 13, 17])
def test_fit_on_sample_sizes_with_integer_quartile_index(n_samples):
    """Exercise the quartile path where the type-7 quartile index is an integer.

    Before the fix this called std::nth_element with nth < first (undefined
    behavior). Release builds of libstdc++ happen to return correct values, so
    this only fails under a checked standard library (e.g. -D_GLIBCXX_DEBUG);
    here it checks that results are well-defined and order-independent.
    """
    rng = np.random.default_rng(n_samples)
    X = np.vstack(
        [
            rng.normal(0, 1, (n_samples // 2, 2)),
            rng.normal(8, 1, (n_samples - n_samples // 2, 2)),
        ]
    )
    init_means = np.array([[0.0, 0.0], [8.0, 8.0]])
    km = KamilaClustering(2, init_means=init_means).fit(X)
    perm = rng.permutation(n_samples)
    km_perm = KamilaClustering(2, init_means=init_means).fit(X[perm])

    assert np.isfinite(km.total_log_lik_)
    np.testing.assert_array_equal(km.labels_[perm], km_perm.labels_)
    assert km.total_log_lik_ == pytest.approx(km_perm.total_log_lik_)
