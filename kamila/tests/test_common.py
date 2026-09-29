import pytest
import sklearn
from sklearn.utils.estimator_checks import _get_check_estimator_ids
from sklearn.utils.fixes import parse_version

from kamila.utils._discovery import all_estimators
from kamila import KamilaClustering

_SKLEARN_VERSION = parse_version(sklearn.__version__)

# Default-constructed estimators treat every feature as continuous, so also check
# a configuration that exercises the categorical code path. (A boolean mask can't
# be used here: the checks fit on varying numbers of features.)
_estimators = [est_cls() for _, est_cls in all_estimators()] + [
    KamilaClustering(categorical_features=[0]),
]


def _expected_failed_checks(estimator):
    """Checks that fail because of bugs in scikit-learn itself."""
    if (  # pragma: no cover
        parse_version("1.6") <= _SKLEARN_VERSION < parse_version("1.7.1")
        and estimator.categorical_features is not None
    ):
        return {
            "check_positive_only_tag_during_fit": (
                "scikit-learn 1.6.0-1.7.0 casts X to int32 for categorical "
                "estimators, then subtracts a float mean in place."
            )
        }
    return {}


try:  # scikit-learn >= 1.6
    from sklearn.utils.estimator_checks import estimator_checks_generator

    def _checks(estimator):
        return estimator_checks_generator(
            estimator,
            legacy=True,
            expected_failed_checks=_expected_failed_checks(estimator),
            mark="xfail",
        )

except ImportError:  # pragma: no cover  (scikit-learn < 1.6)
    from sklearn.utils.estimator_checks import check_estimator

    def _checks(estimator):
        return check_estimator(estimator, generate_only=True)


# Build a list rather than using parametrize_with_checks: on some scikit-learn
# versions it hands pytest a generator, which pytest >= 9.1 deprecates.
@pytest.mark.parametrize(
    "estimator, check",
    [item for est in _estimators for item in _checks(est)],
    ids=_get_check_estimator_ids,
)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API"""
    check(estimator)
