import pytest
from sklearn.utils.estimator_checks import (
    _get_check_estimator_ids,
    estimator_checks_generator,
)

from kamila.utils.discovery import all_estimators

_checks = [
    item
    for _, est_cls in all_estimators()
    for item in estimator_checks_generator(estimator=est_cls(), legacy=True)
    if "array_api" not in getattr(item[1], "func", item[1]).__name__
]


@pytest.mark.parametrize("estimator, check", _checks, ids=_get_check_estimator_ids)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)
