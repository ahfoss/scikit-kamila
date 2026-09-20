"""This file shows how to write tests based on the scikit-learn common tests."""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from kamila.utils.discovery import all_estimators


@pytest.mark.skip(
    reason="Common checks deferred until KamilaClustering algorithm implementation."
)
@parametrize_with_checks([est() for _, est in all_estimators()])
def test_estimators(estimator, check, request):
    """Check the compatibility with scikit-learn API"""
    check(estimator)
