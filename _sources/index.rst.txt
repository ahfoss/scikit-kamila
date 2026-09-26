.. scikit-kamila documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:notoc:

###########################################
scikit-kamila: KAMILA Clustering in Python
###########################################

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/ahfoss/scikit-kamila>`__ |
`Issues & Ideas <https://github.com/ahfoss/scikit-kamila/issues>`__

scikit-kamila is a scikit-learn compatible Python package implementing the
KAMILA (KAy-means for MIxed LArge datasets) clustering algorithm for mixed continuous
and categorical data.


.. grid:: 1 2 3 3
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: User guide
        :img-top: _static/img/index_user_guide.svg
        :class-card: intro-card
        :shadow: md

        The narrative documentation explaining the KAMILA algorithm, how mixed continuous
        and categorical data are modeled, parameter choices, and usage.

        +++

        .. button-ref:: user_guide
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the user guide

    .. grid-item-card:: API reference
        :img-top: _static/img/index_api.svg
        :class-card: intro-card
        :shadow: md

        Detailed descriptions of the ``KamilaClustering`` estimator, its parameters,
        fitted attributes, methods, and utility discovery functions.

        +++

        .. button-ref:: api
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the reference guide

    .. grid-item-card:: Examples
        :img-top: _static/img/index_examples.svg
        :class-card: intro-card
        :shadow: md

        A gallery of examples showcasing clustering of mixed continuous and
        categorical datasets with ``scikit-kamila``.

        +++

        .. button-ref:: general_examples
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the gallery of examples


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    user_guide
    api
    auto_examples/index
