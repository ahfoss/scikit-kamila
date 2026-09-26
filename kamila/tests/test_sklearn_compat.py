"""Regression tests for scikit-learn API conventions (issue #2)."""

import numpy as np
import pytest

from kamila import KamilaClustering
from kamila._validation import _validate_and_split_data


def _mixed(n=20, seed=0):
    rng = np.random.RandomState(seed)
    return np.c_[rng.randint(0, 3, n).astype(float), rng.randn(n)]


def test_non_2d_error_messages():
    with pytest.raises(ValueError, match="got 3D array instead"):
        _validate_and_split_data(np.zeros((4, 2, 2)))
    with pytest.raises(ValueError, match="got 1D array instead"):
        _validate_and_split_data([1.0, 2.0, 3.0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, None])
def test_missing_categorical_values_rejected_in_fit(bad):
    X = _mixed().astype(object)
    X[0, 0] = bad
    with pytest.raises(ValueError, match="contains missing"):
        KamilaClustering(categorical_features=[0], random_state=0).fit(X)


def test_missing_categorical_values_rejected_in_predict():
    X = _mixed()
    kam = KamilaClustering(categorical_features=[0], random_state=0).fit(X)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="contains missing"):
        kam.predict(X)


def test_pandas_missing_categorical_values_rejected():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {"c": pd.array(["a", "b", None, "a"], dtype="string"), "v": [0.0, 1, 2, 3]}
    )
    with pytest.raises(ValueError, match="contains missing"):
        KamilaClustering(categorical_features=["c"], random_state=0).fit(X)


def test_unorderable_categorical_values_raise_type_error():
    X = np.array([["a", 1.0], [1, 2.0], ["b", 3.0]], dtype=object)
    with pytest.raises(TypeError, match=r"argument must be uniformly strings or"):
        KamilaClustering(categorical_features=[0], random_state=0).fit(X)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_weights_rejected(bad):
    X = _mixed()
    with pytest.raises(ValueError, match="con_weights must be finite"):
        KamilaClustering(categorical_features=[0], con_weights=[bad]).fit(X)
    with pytest.raises(ValueError, match="cat_weights must be finite"):
        KamilaClustering(categorical_features=[0], cat_weights=[bad]).fit(X)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"n_clusters": True}, "n_clusters must be an integer"),
        ({"n_init": True}, "n_init must be an integer"),
        ({"max_iter": False}, "max_iter must be an integer"),
        ({"cat_bandwidth": np.nan}, "cat_bandwidth must be a number"),
        ({"cat_bandwidth": True}, "cat_bandwidth must be a number"),
        ({"cat_bandwidth": 1.5}, "cat_bandwidth must be a number"),
    ],
)
def test_invalid_hyperparameters(params, match):
    with pytest.raises(ValueError, match=match):
        KamilaClustering(**params).fit(_mixed())


def test_categorical_tag_follows_categorical_features():
    sklearn_utils = pytest.importorskip("sklearn.utils")
    get_tags = getattr(sklearn_utils, "get_tags", None)
    if get_tags is None:  # pragma: no cover  (scikit-learn < 1.6)
        pytest.skip("get_tags requires scikit-learn >= 1.6")
    assert not get_tags(KamilaClustering()).input_tags.categorical
    assert get_tags(KamilaClustering(categorical_features=[0])).input_tags.categorical


def test_legacy_more_tags():
    assert KamilaClustering()._more_tags() == {"X_types": ["2darray"]}
    assert KamilaClustering(categorical_features="from_dtype")._more_tags() == {
        "X_types": ["2darray", "categorical"]
    }
