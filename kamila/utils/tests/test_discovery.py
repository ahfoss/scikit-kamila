"""Tests for estimator and function discovery utilities."""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

import pytest

from kamila.utils.discovery import all_displays, all_estimators, all_functions


def test_all_estimators():
    estimators = all_estimators()
    assert len(estimators) == 1
    names = [name for name, _ in estimators]
    assert "KamilaClustering" in names

    estimators = all_estimators(type_filter="cluster")
    assert len(estimators) == 1

    estimators = all_estimators(type_filter=["cluster"])
    assert len(estimators) == 1

    estimators = all_estimators(type_filter="classifier")
    assert len(estimators) == 0

    estimators = all_estimators(type_filter="regressor")
    assert len(estimators) == 0

    estimators = all_estimators(type_filter="transformer")
    assert len(estimators) == 0

    estimators = all_estimators(type_filter=["regressor", "transformer"])
    assert len(estimators) == 0

    err_msg = "Parameter type_filter must be"
    with pytest.raises(ValueError, match=err_msg):
        all_estimators(type_filter="xxxx")


def test_is_abstract():
    from abc import ABC, abstractmethod

    from kamila.utils.discovery import _is_abstract

    class ConcreteClass:
        pass

    class EmptyAbstract(ABC):
        pass

    class AbstractWithMethods(ABC):
        @abstractmethod
        def foo(self):
            pass  # pragma: no cover

    assert not _is_abstract(ConcreteClass)
    assert not _is_abstract(EmptyAbstract)
    assert _is_abstract(AbstractWithMethods)


def test_all_displays():
    displays = all_displays()
    assert len(displays) == 0


def test_all_functions():
    functions = all_functions()
    assert len(functions) > 0
    func_names = [name for name, _ in functions]
    assert "all_estimators" in func_names
