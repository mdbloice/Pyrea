# Pyrea: Multi-view hierarchical clustering with flexible ensemble structures
# Copyright (C) 2022 Marcus D. Bloice, Bastian Pfeifer
#
# Licenced under the terms of the MIT license.
"""
The :py:mod:`pyrea.core` module contains all the user-facing API functions
required to use Pyrea. Generally, users will only need to interact with the
functions within this module in order to create their ensemble structures.

Developers, especially those who wish to extend Pyrea, may want to look at the
classes and functions defined in the :py:mod:`pyrea.structure` module.
"""
from array import array
from cmath import exp
from typing import List

from .structure import Agreement, Average, Clusterer, Complete, Disagreement, Ensemble, Fusion, View, Ward, Consensus, Single

CLUSTER_METHODS = ['average', 'complete', 'random', 'single', 'ward']
FUSION_METHODS = ['agreement', 'consensus', 'disagreenent']

def clusterer(clusterer: str, n_clusters: int=2) -> Clusterer:
    """
    Creates a :class:`~pyrea.structure.Clusterer` object to be used when
    creating a :class:`~pyrea.structure.View` or
    :class:`~pyrea.structure.Ensemble`. Can be one of: average, complete,
    random, single, or ward.

    .. code::

        c = pyrea.clusterer('ward', n_clusters=2)

    Then, :attr:`c` can be used when creating a view or executing an ensemble:

    .. code::

        v = pyrea.view(d, c)

    Where :attr:`d` is a data source.

    .. seealso:: The :func:`~view` function.
    .. seealso:: The :func:`~execute_ensemble` function.

    :param clusterer: The type of clusterer to use. Can be one of 'average',
     'complete', 'random', 'single', or 'ward'.
    :param n_clusters: The number of clusters to find. Default=2.
    """
    if not isinstance(clusterer, str):
        raise TypeError("Parameter 'clusterer' must be of type string. Choices available are: %s."
                        % ("'" + "', '".join(CLUSTER_METHODS[:-1]) + "', or '" + CLUSTER_METHODS[-1] + "'"))

    if clusterer not in CLUSTER_METHODS:
        raise TypeError("Parameter 'clusterer' must be one of %s and you passed '%s'."
                        % ("'" + "', '".join(CLUSTER_METHODS[:-1]) + "', or '" + CLUSTER_METHODS[-1] + "'", clusterer))

    if clusterer == 'ward':
        return Ward()
    elif clusterer == 'complete':
        return Complete()
    elif clusterer == 'single':
        return Single()
    elif clusterer == 'average':
        return Average()
    else:
        raise ValueError("Unknown clustering method.")

def view(data: array, clusterer: Clusterer) -> View:
    """
    Creates a :class:`View` object that can subsequently used to create an
    :class:`Ensemble`.

    Views are created using some data in the form of a NumPy matrix or 2D array,
    and a clustering algorithm:

    .. code::

        d = numpy.random.rand(100,10)
        v = pyrea.view(d, c)

    Views are used to create ensembles. They consist of some data, :attr:`d`
    above, and a clustering algorimth, :attr:`c` above.
    """
    return View(data, clusterer)

def fuser(fuser: str):
    """
    Creates a :class:`Fusion` object, which is used to fuse the results of 
    an arbitrarily long list of clusterings.

    .. code::

        f = pyrea.fuser('agreement')

    :param fuser: The fusion algorithm to use. Must be one of 'agreement',
     'disagreement', 'consensus'.
    """
    if not isinstance(fuser, str):
        raise TypeError("Parameter 'fuser' must be of type string.")

    if fuser == "disagreement":
        return Disagreement()
    elif fuser == "agreement":
        return Agreement()
    elif fuser == "consensus":
        return Consensus()


def execute_ensemble(views: List[View], fuser: Fusion, clusterers: List[Clusterer]) -> View:
    """
    Executes an ensemble and returns a new :class:`View` object.

    :param views: The ensemble's views.
    :param fuser: The fusion algorithm used to fuse the clustered data.
    :param clusterers: A clustering algorithm or list of clustering algorithms
     used to cluster the fused matrix created by the fusion algorithm.

    .. code::

        v = pyrea.execute_ensemble([view1, view2, view3], fusion, clusterer)

    Returns a :class:`~pyrea.structure.View` object which can consequently be included in a
    further ensemble.

    .. seealso:: The :func:`~view` function.
    .. seealso:: The :func:`~clusterer` function.

    """
    if not isinstance(views, list):
        raise TypeError("Parameter 'views' must be a list of Views. You provided %s" % type(views))

    return Ensemble(views, fuser, clusterers).execute()

def get_ensemble(views: List[View], fuser: Fusion, clusterers: List[Clusterer]) -> Ensemble:
    """
    Creates and returns an :class:`~pyrea.structure.Ensemble` object which must
    be executed later to get the ensemble's computed view.
    """
    if not isinstance(views, list):
        raise TypeError("Parameter 'views' must be a list of Views. You provided %s" % type(views))

    return Ensemble(views, fuser, clusterers)

def summary():
    """
    Not yet implemented.

    Prints a summary of the current ensemble structure, including any
    already calculated statistics.
    """
    title = "Summary Statistics"
    print(f" {title.title()} ".center(80, '*'))

    print("\n")
    print(f"Not yet implemented".center(80))
    print("\n")

    print(f" End Summary ".center(80, "*"))
