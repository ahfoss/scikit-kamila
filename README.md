# scikit-kamila

[![tests](https://github.com/ahfoss/scikit-kamila/actions/workflows/tests.yml/badge.svg)](https://github.com/ahfoss/scikit-kamila/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ahfoss/scikit-kamila/branch/main/graph/badge.svg)](https://codecov.io/gh/ahfoss/scikit-kamila)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/ahfoss/scikit-kamila)
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
from kamila import KamilaClustering

# Initialize the estimator
kamila = KamilaClustering(n_clusters=2, random_state=42)
# TODO
```

## Development & Contributing

### Code Coverage Requirement
All pull requests and contributions must maintain **100% test code coverage**. Continuous integration enforces this requirement via automated test workflows across Python 3.9 through 3.15.

To run the test suite and verify 100% coverage locally:

```bash
# Run pytest with missing line reporting and 100% threshold enforcement
python -m pytest --cov=kamila --cov-report=term-missing --cov-fail-under=100
```

## References

* [Foss, Markatou, Ray, and Heching (2016). A semiparametric method for clustering mixed data. **Machine Learning**, 105(3), 419-458. DOI: 10.1007/s10994-016-5575-7](https://link.springer.com/article/10.1007/s10994-016-5575-7)
* [Foss and Markatou (2018). kamila: Clustering Mixed-Type Data in R and Hadoop. **Journal of Statistical Software**, 83(13). DOI: 10.18637/jss.v083.i13](https://www.jstatsoft.org/article/view/v083i13)
* [Foss, Markatou, and Ray (2018). Distance Metrics and Clustering Methods for Mixed-Type Data. **International Statistical Review**. DOI: 10.1111/insr.12274.](https://onlinelibrary.wiley.com/doi/abs/10.1111/insr.12274)
- Original R package: [ahfoss/kamila](https://github.com/ahfoss/kamila).

