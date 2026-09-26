import pytest

# `estimator_checks_generator` was added in scikit-learn 1.6. Older supported
# versions skip this module instead of failing collection of the whole suite.
pytest.importorskip("sklearn", minversion="1.6")

from sklearn.utils.estimator_checks import (  # noqa: E402
    _get_check_estimator_ids,
    estimator_checks_generator,
)

from kamila.utils.discovery import all_estimators  # noqa: E402

# Array API checks are excluded: KamilaClustering sets `array_api_support=False`,
# and these checks would otherwise be reported as skipped.
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
