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
import numpy as np

from .structure import Agreement, Clusterer, DBSCANPyrea, Disagreement
from .structure import Ensemble, Fusion, HierarchicalClusteringPyrea
from .structure import OPTICSPyrea, SpectralClusteringPyrea, View, Consensus

CLUSTER_METHODS = ['spectral', 'hierarchical', 'dbscan', 'optics']
FUSION_METHODS = ['agreement', 'consensus', 'disagreement']
LINKAGES = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

def clusterer(clusterer: str, precomputed: bool=False, **kwargs) -> Clusterer:
    """
    Creates a :class:`~pyrea.structure.Clusterer` object to be used when
    creating a :class:`~pyrea.structure.View` or
    :class:`~pyrea.structure.Ensemble`. Can be one of: :attr:`'spectral'`,
    :attr:`'hierarchical'`, :attr:`'dbscan'`, or :attr:`'optics'`.

    .. code::

        c = pyrea.clusterer('hierarchical', n_clusters=2)

    Then, :attr:`c` can be used when creating a view:

    .. code::

        v = pyrea.view(d, c)

    Where :attr:`d` is a data source.

    .. seealso:: The :func:`~view` function.
    .. seealso:: The :func:`~execute_ensemble` function.

    Each clustering algorithm has a different set of parameters, default values
    are used throughout and can be overridden if required. For example,
    hierarchical and spectral clustering allow you to specify the number of
    clusters to find using :attr:`n_clusters`, while DBSCAN and OPTICS do not.

    Also, hierarchical clustering allows for a :attr:`distance_metric` to be
    set, which can be one of: :attr:`'braycurtis'`, :attr:`'canberra'`,
    :attr:`'chebyshev'`, :attr:`'cityblock'`, :attr:`'correlation'`,
    :attr:`'cosine'`, :attr:`'dice'`, :attr:`'euclidean'`, :attr:`'hamming'`,
    :attr:`'jaccard'`, :attr:`'jensenshannon'`, :attr:`'kulczynski1'`,
    :attr:`'mahalanobis'`, :attr:`'matching'`, :attr:`'minkowski'`,
    :attr:`'rogerstanimoto'`, :attr:`'russellrao'`, :attr:`'seuclidean'`,
    :attr:`'sokalmichener'`, :attr:`'sokalsneath'`, :attr:`'sqeuclidean'`, or
    :attr:`'yule'`.

    Likewise, adjusting the linkage method is possible using hierarchical
    clustering algorithms, this can be one of: :attr:`'single'`,
    :attr:`'complete'`, :attr:`'average'`, :attr:`'weighted'`,
    :attr:`'centroid'`, :attr:`'median'`, or :attr:`'ward'`.

    For complete documentation of each clustering algorithm's parameters see
    the following:

    * Spectral: :class:`~pyrea.structure.SpectralClusteringPyrea`
    * Hierarchical: :class:`~pyrea.structure.HierarchicalClusteringPyrea`
    * DBSCAN: :class:`~pyrea.structure.DBSCANPyrea`
    * OPTICS: :class:`~pyrea.structure.OPTICSPyrea`

    :param clusterer: The type of clusterer to use. Can be one of:
     :attr:`'spectral'`, :attr:`'hierarchical'`, :attr:`'dbscan'`,
     or :attr:`'optics'`.
    :param precomputed: Whether the clusterer should assume the data is a
     distance matrix.
    :param \*\*kwargs: Keyword arguments to be passed to the clusterer.
     See each clustering algorithm's documentation for full details: Spectral:
     :class:`~pyrea.structure.SpectralClusteringPyrea`, Hierarchical:
     :class:`~pyrea.structure.HierarchicalClusteringPyrea`, DBSCAN:
     :class:`~pyrea.structure.DBSCANPyrea`, and OPTICS:
     :class:`~pyrea.structure.OPTICSPyrea`.
    """
    if not isinstance(clusterer, str):
        raise TypeError("Parameter 'clusterer' must be of type string. Choices available are: %s."
                        % ("'" + "', '".join(CLUSTER_METHODS[:-1]) + "', or '" + CLUSTER_METHODS[-1] + "'"))

    if clusterer not in CLUSTER_METHODS:
        raise TypeError("Parameter 'clusterer' must be one of %s: you passed '%s'."
                        % ("'" + "', '".join(CLUSTER_METHODS[:-1]) + "', or '" + CLUSTER_METHODS[-1] + "'", clusterer))

    if clusterer == 'spectral':
        if precomputed:
            kwargs['affinity'] = 'precomputed'

        return SpectralClusteringPyrea(**kwargs)

    elif clusterer == 'hierarchical':

        if kwargs['method']:
            if kwargs['method'] not in LINKAGES:
                raise TypeError("Illegal method.")

        return HierarchicalClusteringPyrea(precomputed=precomputed, **kwargs)

    elif clusterer == 'dbscan':
        if precomputed:
            kwargs['metric']='precomputed'

        return DBSCANPyrea(**kwargs)

    elif clusterer == 'optics':
        if precomputed:
            kwargs['metric']='precomputed'

        return OPTICSPyrea(**kwargs)

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


def execute_ensemble(views: List[View], fuser: Fusion) -> list:
    """
    Executes an ensemble and returns a new :class:`View` object.

    :param views: The ensemble's views.
    :param fuser: The fusion algorithm used to fuse the clustered data.
    :param clusterers: A clustering algorithm or list of clustering algorithms
     used to cluster the fused matrix created by the fusion algorithm.

    .. code::

        v = pyrea.execute_ensemble([view1, view2, view3], fusion, clusterer)

    Returns a :class:`~pyrea.structure.View` object which can consequently be
    included in a further ensemble.

    .. seealso:: The :func:`~view` function.
    .. seealso:: The :func:`~clusterer` function.

    """
    if not isinstance(views, list):
        raise TypeError("Parameter 'views' must be a list of Views. You provided %s" % type(views))

    return Ensemble(views, fuser).execute()

def get_ensemble(views: List[View], fuser: Fusion, clusterers: List[Clusterer]) -> Ensemble:
    """
    Creates and returns an :class:`~pyrea.structure.Ensemble` object which must
    be executed later to get the ensemble's computed view.
    """
    if not isinstance(views, list):
        raise TypeError("Parameter 'views' must be a list of Views. You provided %s" % type(views))

    return Ensemble(views, fuser, clusterers)

def consensus(labels: list):

    if len(labels) <= 1:
        raise ValueError("You must provide a list of labellings of length >= 2.")

    n_samp    = len(labels[0])
    cl_cons   = np.zeros((n_samp,), dtype=int)
    n_cl = len(labels)

    k = 1

    for i in range(0, n_samp):

        ids = np.where(labels[0] == labels[0][i])

        for j in range(1, n_cl):

            m = np.where(labels[j] == labels[j][i])
            ids = np.intersect1d(ids, m)

        check = np.sum(cl_cons[ids])
        if check == 0:
            cl_cons[ids] = k
            k = k + 1

    return cl_cons

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
