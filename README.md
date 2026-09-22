# scikit-kamila

[![tests](https://github.com/ahfoss/scikit-kamila/actions/workflows/tests.yml/badge.svg)](https://github.com/ahfoss/scikit-kamila/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ahfoss/scikit-kamila/branch/main/graph/badge.svg)](https://codecov.io/gh/ahfoss/scikit-kamila)
[![codecov (c++)](https://codecov.io/gh/ahfoss/scikit-kamila/branch/main/graph/badge.svg?flag=cpp)](https://codecov.io/gh/ahfoss/scikit-kamila)
[![C++ Coverage](https://img.shields.io/badge/C%2B%2B_Coverage-100%25-brightgreen.svg)](https://github.com/ahfoss/scikit-kamila)
[![doc](https://github.com/ahfoss/scikit-kamila/actions/workflows/docs.yml/badge.svg)](https://ahfoss.github.io/scikit-kamila)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**scikit-kamila** is a scikit-learn-compatible Python package implementing **KAMILA** (KAy-means for MIxed LArge datasets), an iterative clustering algorithm designed for mixed-type data (combinations of continuous and categorical variables).

## Installation

```bash
pip install kamila
```

## Quick Start

```python
from kamila import KamilaClustering

# Initialize the estimator
kamila = KamilaClustering(n_clusters=2, random_state=42)
```

## Development & Contributing

### Code Coverage Requirement
All pull requests and contributions must maintain **100% test code coverage** across both Python and C++ source code. Continuous integration enforces this requirement via automated test workflows across Python 3.9 through 3.15.

To run the test suite and verify 100% Python coverage locally:

```bash
# Run pytest with missing line reporting and 100% threshold enforcement
python -m pytest --cov=kamila --cov-report=term-missing --cov-fail-under=100
```

To run and generate C++ code coverage reports:

```bash
# Build with coverage and generate C++ report via gcovr or OpenCppCoverage
python scripts/run_cpp_coverage.py --html --xml
```

## References

- Foss A, Markatou M. (2018). *kamila: Clustering Mixed-Type Data in R and Hadoop*. Journal of Statistical Software, 83(13), 1–45. doi: [10.18637/jss.v083.i13](https://doi.org/10.18637/jss.v083.i13).
- Original R package: [ahfoss/kamila](https://github.com/ahfoss/kamila).

