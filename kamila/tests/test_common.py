from sklearn.utils.estimator_checks import parametrize_with_checks

from kamila import KamilaClustering
from kamila.utils.discovery import all_estimators

# Default-constructed estimators treat every feature as continuous, so also check
# a configuration that exercises the categorical code path. (A boolean mask can't
# be used here: the checks fit on varying numbers of features.)
_estimators = [est_cls() for _, est_cls in all_estimators()] + [
    KamilaClustering(categorical_features=[0]),
]


# parametrize_with_checks (rather than estimator_checks_generator, which is new in
# scikit-learn 1.6) keeps this test runnable on the oldest supported version.
@parametrize_with_checks(_estimators)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)
