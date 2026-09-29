from sklearn.utils.estimator_checks import parametrize_with_checks

from kamila.utils.discovery import all_estimators


@parametrize_with_checks([est_cls() for _, est_cls in all_estimators()])
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)
