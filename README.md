# scikit-kamila

[![tests](https://github.com/ahfoss/scikit-kamila/actions/workflows/tests.yml/badge.svg)](https://github.com/ahfoss/scikit-kamila/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ahfoss/scikit-kamila/branch/main/graph/badge.svg)](https://codecov.io/gh/ahfoss/scikit-kamila)
[![codecov (c++)](https://codecov.io/gh/ahfoss/scikit-kamila/branch/main/graph/badge.svg?flag=cpp)](https://codecov.io/gh/ahfoss/scikit-kamila)
[![C++ Coverage](https://img.shields.io/badge/C%2B%2B_Coverage-100%25-brightgreen.svg)](https://github.com/ahfoss/scikit-kamila)
[![doc](https://github.com/ahfoss/scikit-kamila/actions/workflows/docs.yml/badge.svg)](https://ahfoss.github.io/scikit-kamila)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**scikit-kamila** is a scikit-learn-compatible Python package implementing **KAMILA** (KAy-means for MIxed LArge datasets), an iterative clustering algorithm designed for mixed-type data (combinations of continuous and categorical variables).

Continuous variables are treated similarly to *k*-means in which clusters are modeled as spherical, although this technique relaxes these assumptions by merely assuming the clusters arise from distributions with spherical contour lines.

Categorical variables are treated as manifestations from multinomial distributions.

## Installation

```bash
pip install kamila
```

## Quick Start

```python
import pandas as pd
from kamila import KamilaClustering

X = pd.DataFrame(
    {
        "height": [1.1, 0.9, 1.0, 5.2, 4.8, 5.1],
        "weight": [2.0, 2.2, 1.9, 8.1, 7.9, 8.3],
        "color": pd.Categorical(["red", "red", "blue", "green", "green", "blue"]),
    }
)

# Columns with a pandas "category" (or bool) dtype are treated as categorical;
# alternatively pass column indices, names, or a boolean mask.
kamila = KamilaClustering(n_clusters=2, categorical_features="from_dtype", random_state=42)
labels = kamila.fit_predict(X)

kamila.cluster_centers_con_  # continuous cluster centers, shape (2, 2)
kamila.cluster_centers_cat_  # per-feature categorical log-probabilities
kamila.predict(X.head(2))    # assign new data to the fitted clusters
```

`KamilaClustering` follows the scikit-learn estimator API, so it can be used in
pipelines, cloned, and tuned with `GridSearchCV` (pass an explicit `scoring`, such
as `"adjusted_rand_score"`, since it does not define a `score` method).

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

* [Foss, Markatou, Ray, and Heching (2016). A semiparametric method for clustering mixed data. **Machine Learning**, 105(3), 419-458. DOI: 10.1007/s10994-016-5575-7](https://link.springer.com/article/10.1007/s10994-016-5575-7)
* [Foss and Markatou (2018). kamila: Clustering Mixed-Type Data in R and Hadoop. **Journal of Statistical Software**, 83(13). DOI: 10.18637/jss.v083.i13](https://www.jstatsoft.org/article/view/v083i13)
* [Foss, Markatou, and Ray (2018). Distance Metrics and Clustering Methods for Mixed-Type Data. **International Statistical Review**. DOI: 10.1111/insr.12274.](https://onlinelibrary.wiley.com/doi/abs/10.1111/insr.12274)
- Original R package: [ahfoss/kamila](https://github.com/ahfoss/kamila).

