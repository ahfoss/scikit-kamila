.. title:: User guide : contents

.. _user_guide:

==========
User Guide
==========

The ``kamila`` package provides a scikit-learn compatible implementation of **KAMILA**
(KAy-means for MIxed LArge datasets), an iterative clustering algorithm designed
specifically for mixed-type datasets containing both continuous and categorical variables.

Clustering Mixed-Type Data
--------------------------

Clustering datasets with mixed continuous and categorical variables presents unique
methodological challenges:

* **Scale and Metric Incompatibility**: Continuous variables naturally live in metric
  spaces (such as Euclidean space), whereas categorical variables represent discrete,
  unordered or ordered qualitative states where Euclidean distances are not directly
  meaningful.
* **Limitations of Standard Heuristics**:

  * *Dummy variable k-means*: One-hot encoding categorical variables and running standard
    k-means distorts the geometry of the space and artificially inflates dimensionality.
  * *Gower distance with k-medoids (PAM)*: While distance-based approaches like Gower's
    coefficient accommodate mixed types, pairwise distance computation and medoid updates
    scale quadratically :math:`\mathcal{O}(N^2)` with sample size :math:`N`, making them
    prohibitive for large datasets.
  * *Parametric mixture models*: Latent class / Gaussian mixture models require rigid
    distributional assumptions that may not hold in practice.

The KAMILA Algorithm
--------------------

KAMILA (Foss and Markatou, 2018) is a semi-parametric clustering method that resolves these
issues:

1. **Continuous Domain**: Continuous variables are modeled using non-parametric radial
   kernel density estimation around cluster centroids.
2. **Categorical Domain**: Categorical variables are modeled using multinomial distributions
   with additive Laplace/Dirichlet smoothing controlled by a bandwidth parameter.
3. **Dynamic Weighting**: KAMILA dynamically balances the continuous and categorical
   contributions during the assignment step using probability densities rather than
   requiring manual ad-hoc weight tuning.
4. **Computational Efficiency**: An optimized C++ computational core provides fast
   convergence, scaling linearly :math:`\mathcal{O}(N \cdot K)` per iteration with sample
   size :math:`N` and number of clusters :math:`K`.

The KamilaClustering Estimator
------------------------------

The primary interface in ``scikit-kamila`` is the :class:`~kamila.KamilaClustering`
estimator, which inherits from scikit-learn's :class:`sklearn.base.BaseEstimator` and
:class:`sklearn.base.ClusterMixin`.

Basic Usage
~~~~~~~~~~~

To perform clustering on a mixed dataset:

.. code-block:: python

    import numpy as np
    from kamila import KamilaClustering

    # Continuous features (2 columns)
    rng = np.random.RandomState(42)
    X_con = np.vstack([rng.randn(50, 2), rng.randn(50, 2) + 3.0])

    # Categorical features (1 column: categories 'cat_A' and 'cat_B')
    X_cat = np.array([['cat_A']] * 50 + [['cat_B']] * 50)

    # Combined mixed dataset
    X = np.hstack([X_con, X_cat])

    # Initialize and fit KAMILA with 2 clusters, indicating column 2 is categorical
    clusterer = KamilaClustering(n_clusters=2, categorical_features=[2], random_state=42)
    clusterer.fit(X)

    # Obtain cluster assignments
    labels = clusterer.labels_

Specifying Categorical Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``categorical_features`` parameter provides flexible ways to specify which
columns should be treated as categorical:

* ``None`` (default): All features are treated as continuous.
* ``'from_dtype'``: Automatically inspects Pandas DataFrames and treats columns with
  ``category`` or ``bool`` dtypes as categorical.
* **Integer indices**: A list or array of column indices, such as ``[2, 3]`` or ``[-1]``.
* **Feature names**: A list of column name strings, such as ``['region', 'tier']``
  (when ``X`` is a DataFrame or has feature names).
* **Boolean mask**: A boolean array of shape ``(n_features,)``, where ``True`` indicates
  a categorical feature.

Key Hyperparameters
~~~~~~~~~~~~~~~~~~~

* ``n_clusters`` (*int*, default=2):
  The number of clusters :math:`K` to form.
* ``categorical_features`` (*None, 'from_dtype', or array-like*, default=None):
  Specification of categorical columns.
* ``n_init`` (*int*, default=10):
  Number of random initializations. The best run (lowest objective value / inertia) is
  retained.
* ``max_iter`` (*int*, default=25):
  Maximum number of iterations allowed per initialization.
* ``cat_bandwidth`` (*float*, default=0.025):
  Categorical smoothing parameter :math:`\delta \in [0, 1]`. Keep it small: see
  :ref:`cat_bandwidth_choice` below.
* ``con_weights`` (*array-like of shape (n_con,)*, default=None):
  Optional per-feature weights for continuous variables.
* ``cat_weights`` (*array-like of shape (n_cat,)*, default=None):
  Optional per-feature weights for categorical variables.
* ``random_state`` (*int, RandomState instance, or None*, default=None):
  Determines random number generation for centroid initializations.

Fitted Attributes
~~~~~~~~~~~~~~~~~

After fitting, :class:`~kamila.KamilaClustering` provides standard scikit-learn
attributes:

* ``cluster_centers_con_``: Coordinates of continuous cluster centroids of shape
  ``(n_clusters, n_con)`` (or ``None`` if no continuous features).
* ``cluster_centers_cat_``: List of log-probability matrices for each categorical feature
  (or ``None`` if no categorical features).
* ``labels_``: Cluster label for each sample in the training data of shape ``(n_samples,)``.
* ``inertia_``: Objective value of the winning initialization.
* ``n_iter_``: Number of iterations taken by the winning initialization.
* ``n_features_in_``: Total number of features seen during :meth:`fit`.
* ``feature_names_in_``: Names of features seen during :meth:`fit` (if provided).
* ``is_categorical_``: Boolean mask indicating categorical features.
* ``categories_``: List of unique levels observed for each categorical feature.
* ``num_levels_``: Array containing the number of distinct levels per categorical feature.

Predicting on New Data
~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~kamila.KamilaClustering.predict` method assigns new data points to the closest
cluster according to the learned continuous centers and categorical probability distributions:

.. code-block:: python

    new_X = np.array([[0.1, -0.2, 'cat_A'], [3.2, 2.9, 'cat_B']], dtype=object)
    predictions = clusterer.predict(new_X)

Integration with Scikit-Learn Pipelines
---------------------------------------

Because :class:`~kamila.KamilaClustering` complies with the scikit-learn API, it integrates
seamlessly with standard scikit-learn workflows and evaluation metrics:

.. code-block:: python

    import pandas as pd
    from kamila import KamilaClustering
    from sklearn.metrics import adjusted_rand_score

    df = pd.DataFrame({
        'age': [25, 47, 31, 54, 23, 60],
        'income': [50000.0, 110000.0, 62000.0, 130000.0, 48000.0, 140000.0],
        'education': pd.Series(['BSc', 'PhD', 'BSc', 'PhD', 'BSc', 'PhD'], dtype='category'),
    })

    kamila = KamilaClustering(n_clusters=2, categorical_features='from_dtype', random_state=42)
    labels = kamila.fit_predict(df)

Mathematical Details
--------------------

For a sample :math:`\mathbf{x}_i = (\mathbf{x}_i^{(con)}, \mathbf{x}_i^{(cat)})` and cluster
center :math:`k`, the continuous distance is given by weighted Euclidean distance:

.. math::

    d_{ik}^2 = \sum_{j=1}^{P_{con}} w_j^{(con)} \left(x_{ij}^{(con)} - \mu_{kj}^{(con)}\right)^2

The continuous density contribution is evaluated via a radial kernel density estimator
over the continuous distances:

.. math::

    f_{con}(\mathbf{x}_i^{(con)} \mid k) = \frac{1}{N_k h_k^{P_{con}}} \sum_{m \in C_k} K\left(\frac{\|\mathbf{x}_i^{(con)} - \mathbf{x}_m^{(con)}\|}{h_k}\right)

The categorical probability is modeled as the product of independent multinomial probabilities
smoothed by categorical bandwidth :math:`\delta`:

.. math::

    \log P(\mathbf{x}_i^{(cat)} \mid k) = \sum_{j=1}^{P_{cat}} w_j^{(cat)} \log \tilde{p}_{k j}(x_{ij}^{(cat)})

The smoothed counts :math:`\tilde{n}` behind :math:`\tilde{p}` are computed in two passes,
first across the :math:`K` clusters and then across the :math:`L_j` levels of the
variable. Each pass keeps weight :math:`1 - \delta` on a cell and spreads weight
:math:`\delta / (m - 1)` to each of the other :math:`m - 1` cells, with
:math:`m = K` or :math:`m = L_j`:

.. math::

    \tilde{n}_{c} = (1 - \delta)\, n_{c} + \frac{\delta}{m - 1} \sum_{c' \neq c} n_{c'}

.. _cat_bandwidth_choice:

Choosing ``cat_bandwidth``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Smoothing only makes sense while a cell outweighs each of its neighbors, i.e. while
:math:`1 - \delta \geq \delta / (m - 1)`, which is equivalent to

.. math::

    \delta \leq \frac{m - 1}{m}.

At larger values a category's smoothed count is influenced more by each neighboring
cluster (or level) than by itself. :meth:`~kamila.KamilaClustering.fit` issues a
``UserWarning`` when :math:`\delta > (m - 1)/m` for :math:`m = K` or for the number of
levels of any categorical variable. For example, with two clusters or a binary
variable the bound is :math:`0.5`. Small values such as the default of 0.025 are
recommended.

Points are iteratively reassigned to minimize the combined objective criterion until
cluster assignments stabilize.

References
----------

* Foss, A., & Markatou, M. (2018). *kamila: Clustering Mixed-Type Data in R and Hadoop*.
  Journal of Statistical Software, 83(13), 1–45. doi: `10.18637/jss.v083.i13 <https://doi.org/10.18637/jss.v083.i13>`_.
* Foss, A., Markatou, M., & Ray, S. (2016). *A semiparametric method for clustering mixed data*.
  Machine Learning, 105(3), 419–458. doi: `10.1007/s10994-016-5575-5 <https://doi.org/10.1007/s10994-016-5575-5>`_.
