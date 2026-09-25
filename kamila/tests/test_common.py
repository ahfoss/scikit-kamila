import pytest
from sklearn.utils.estimator_checks import (
    _get_check_estimator_ids,
    estimator_checks_generator,
)

from kamila.utils.discovery import all_estimators

_checks = []
for _, est_cls in all_estimators():
    est = est_cls()
    for item in estimator_checks_generator(estimator=est, legacy=True):
        check_fn = item[1]
        check_name = getattr(check_fn, "func", check_fn).__name__
        if "array_api" in check_name:
            continue
        _checks.append(item)


@pytest.mark.parametrize("estimator, check", _checks, ids=_get_check_estimator_ids)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)


