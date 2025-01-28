# Pyrea: Multi-view clustering with deep ensemble structures
# Copyright (C) 2022 Marcus D. Bloice, Bastian Pfeifer
#
# Licenced under the terms of the MIT license.
#
# structure.py
# Contains the classes required for the structuring of ensembles, for
# example Views, Ensembles, Clusterers, and so on.
"""
The :py:mod:`pyrea.structure` module contains the classes used for the internal functionaliy
of Pyrea. The classes contained here are not generally called or instantiated
by the user, see the :py:mod:`pyrea.core` module for the user-facing API.

Developers who wish to extend Pyrea, such as by creating a custom clustering
algorthim, should consult the documentation of the :class:`Clusterer` abstract
base class for example. The :class:`Fusion` class is another such abstract base
class that must be used if a developer wishes to create a custom fusion
algorithm for use within Pyrea.
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN, OPTICS
from typing import List, Union, Any
from scipy.cluster import hierarchy
from scipy import spatial

class Clusterer(object):
    """
    :class:`Clusterer` is the Abstract Base Class for all clustering algorithms.
    All clustering algorithms must be a subclass of this class in order to
    accepted by functions such as :func:`~pyrea.core.execute_ensemble()`.
    To extend Pyrea with a custom clustering algorithm, create a new
    class that is a subclass of :class:`Clusterer`, and implement the
    :func:`Clusterer.execute` function.
    """
    def __init__(self) -> None:
        pass

    def execute(self) -> list:
        """
        Execute the clustering algorithm with the given :attr:`data`.
        """
        pass


class HierarchicalClusteringPyrea(Clusterer):
    def __init__(self, precomputed,
                       # linkage arguments:
                       method='single',
                       metric='euclidean',
                       optimal_ordering=False,
                       # pdist arguments:
                       distance_metric = 'euclidean',
                       out=None,
                       # cut_tree arguments:
                       n_clusters=None,
                       height=None
                       ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.optimal_ordering = optimal_ordering
        self.distance_metric = distance_metric
        self.precomputed = precomputed
        self.out = out
        self.height = height

    def execute(self, data) -> list:
        super().execute()

        # TODO: Both pdist and linkage can take a distance metric.
        # It must be possible for the user to provide both.
        # Currently distance_metric is used for pdist although this then
        # breaks compatibility with the SciPy docs. Fix.

        if self.precomputed:
            y = spatial.distance.squareform(data)

            if self.method == 'ward2':
                y = y**2

            tree = hierarchy.linkage(y, method='ward', metric=self.metric)

        else:
            y = spatial.distance.pdist(data, metric=self.distance_metric, out=self.out)

            if self.method == 'ward2':
                y = y**2

            tree = hierarchy.linkage(y, method='ward', metric=self.metric)

        return hierarchy.cut_tree(tree, n_clusters=self.n_clusters, height=self.height)


class AgglomerativeClusteringPyrea(Clusterer):
    def __init__(self, n_clusters=2,
                       linkage: str='ward',
                       affinity: str='euclidean',
                       memory: Union[None, Any]=None,
                       connectivity=None,
                       compute_full_tree='auto',
                       distance_threshold=None,
                       compute_distances=False) -> None:
        """
        Perform agglomerative clustering.


        See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

        """
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances

    def execute(self, data: list) -> list:
        super().execute()
        return AgglomerativeClustering(n_clusters = self.n_clusters,
                                       linkage=self.linkage,
                                       affinity=self.affinity,
                                       memory=self.memory,
                                       connectivity=self.connectivity,
                                       compute_full_tree=self.compute_full_tree,
                                       distance_threshold=self.distance_threshold,
                                       compute_distances=self.compute_distances).fit(data).labels_


class SpectralClusteringPyrea(Clusterer):
    def __init__(self, n_clusters=8,
                       eigen_solver=None,
                       n_components=None,
                       random_state=None,
                       n_init=10,
                       gamma=1.0,
                       affinity='nearest_neighbors',
                       n_neighbors=10,
                       eigen_tol=0.0,
                       assign_labels='kmeans',
                       degree=3,
                       coef0=1,
                       kernel_params=None,
                       n_jobs=None,
                       verbose=False,
                       method=None) -> None:  # method is not used, but is here
                                              # for compatibility with other
                                              # clustering algorithms
        """
        Perform spectral clustering.

        See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def execute(self, data: list) -> list:
        super().execute()

        return SpectralClustering(n_clusters=self.n_clusters,
                                    eigen_solver=self.eigen_solver,
                                    n_components=self.n_components,
                                    random_state=self.random_state,
                                    n_init=self.n_init,
                                    gamma=self.gamma,
                                    affinity=self.affinity,
                                    n_neighbors=self.n_neighbors,
                                    eigen_tol=self.eigen_tol,
                                    assign_labels=self.assign_labels,
                                    degree=self.degree,
                                    coef0=self.coef0,
                                    kernel_params=self.kernel_params,
                                    n_jobs=self.n_jobs,
                                    verbose=self.verbose).fit(data).labels_


class DBSCANPyrea(Clusterer):
    def __init__(self, eps=0.5,
                       min_samples=5,
                       metric='euclidean',
                       metric_params=None,
                       algorithm='auto',
                       leaf_size=30,
                       p=None,
                       n_jobs=None,
                       n_clusters=None,
                       method=None, # n_clusters and method are not used,
                                    # but are here for compatibility with other
                                    # clustering algorithms
                       ) -> None:
        super().__init__()

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def execute(self, data) -> list:
        DBSCAN(eps=self.eps,
               min_samples=self.min_samples,
               metric=self.metric,
               metric_params=self.metric_params,
               algorithm=self.algorithm,
               leaf_size=self.leaf_size,
               p=self.p,
               n_jobs=self.n_jobs).fit(data).labels_


class OPTICSPyrea(Clusterer):
    def __init__(self, min_samples=5,
                       max_eps=np.inf,
                       metric='minkowski',
                       p=2,
                       metric_params=None,
                       cluster_method='xi',
                       eps=None,
                       xi=0.05,
                       predecessor_correction=True,
                       min_cluster_size=None,
                       algorithm='auto',
                       leaf_size=30,
                       # memory=None,
                       n_jobs=None,
                       n_clusters=None,
                       method=None) -> None:  # n_clusters and method are not
                                              # used, but are here for
                                              # compatibility with other
                                              # clustering algorithms
        super().__init__()
        self.max_eps = max_eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.leaf_size = leaf_size
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        # self.memory = memory  # TODO: This must be a new parameter since some version. Check.
        self.n_jobs = n_jobs

    def execute(self, data: list) -> list:

        return OPTICS(max_eps=self.max_eps,
                      min_samples=self.min_samples,
                      min_cluster_size=self.min_cluster_size,
                      algorithm=self.algorithm,
                      metric=self.metric,
                      metric_params=self.metric_params,
                      p=self.p,
                      leaf_size=self.leaf_size,
                      cluster_method=self.cluster_method,
                      eps=self.eps,
                      xi=self.xi,
                      predecessor_correction=self.predecessor_correction,
                      # memory = self.memory,
                      n_jobs=self.n_jobs
                      ).fit(data).labels_


class Fusion(object):
    def __init__(self) -> None:
        """
        :class:`Fusion` is the Abstract Base Class for all fusion algorithms.
        All fusion algorithms must be a subclass of this class in order to
        accepted by functions such as :func:`~pyrea.core.execute_ensemble()`.
        To extend Pyrea with a custom fusion algorithm, create a new
        class that is a subclass of :class:`Fusion`, and implement the
        :func:`Fusion.execute` function.
        """
        pass

    def execute(self, views: list) -> list:
        """
        Execute the fusion algorithm on the provided :attr:`views`.
        """
        # TODO: Fix views type to List[View] (requires reshuffle of class order)
        pass


class Parea(Fusion):
    """
    Parea fusion algorithm. This functionality is not yet implemented.
    """
    def __init__(self) -> None:
        super().__init__()
        raise Exception("Not yet implemented.")

    def execute(self, views: list) -> list:
        """
        Performs the fusion of a set of views.

        Not yet implemented.
        """
        # TODO: Check name, is it HCfused?
        pass


class Disagreement(Fusion):
    """
    Disagreement fusion function.

    Creates the disagreement of two clusterings.
    """
    def __init__(self) -> None:
        super().__init__()

    def execute(self, views: list) -> list:
        """
        Executes the disagreement fusion algorithm on the provided clusterings,
        :attr:`views`.
        """
        n = len(views[0])
        labels = np.zeros((n, n), dtype=int)

        for i in range(0, len(views)):
            l = views[i]
            res = [[int(x != y) for y in l] for x in l]
            res = np.array(res)
            labels = labels + res

        return labels


class Agreement(Fusion):
    """
    Agreement fusion function.

    Creates the agreement of two clusterings.
    """
    def __init__(self) -> None:
        super().__init__()

    # TODO: Rename paramter to labels
    def execute(self, views: list) -> list:
        """
        Executes the agreement fusion algorithm on the provided clusterings,
        :attr:`views`.
        """
        n_samp  = len(views[0])

        labels  = np.zeros((n_samp, n_samp), dtype=int)

        for i in range(0, len(views)):

            l = views[i]
            res = [[int(x == y) for y in l] for x in l]
            res = np.array(res)
            labels = labels + res

        # in_place=False does not work, we have to edit the matrix in place
        # labels = np.fill_diagonal(labels, 0, in_place=False)
        np.fill_diagonal(labels, 0)
        return labels


class Consensus(Fusion):
    """
    Consensus fusion function.

    Creates the consensus of two clusterings.
    """
    def __init__(self) -> None:
        super().__init__()

    def execute(self, views: list):
        """
        Executes the consensus fusion algorithm on the provided clusterings,
        :attr:`views`.
        """
        # Start consensus
        n_samp    = len(views[0])
        cl_cons   = np.zeros((n_samp,), dtype=int)

        n_cl = len(views)

        k = 1
        for xx in range(0, n_samp):

            ids = np.where(views[0] == views[0][xx])

            for yy in range(1, n_cl):

                m = np.where(views[yy] == views[yy][xx])
                ids = np.intersect1d(ids, m)

            check = np.sum(cl_cons[ids])
            if check == 0:
                cl_cons[ids] = k
                k = k + 1
        # End consensus

        # Calculate binary matrix
        mat_bin   = np.zeros((n_samp, n_samp), dtype=int)
        for xx in range(0, n_samp):

            ids = np.where(cl_cons == cl_cons[xx])
            mat_bin[xx, ids] = 1
            mat_bin[ids, xx] = 1

        return(mat_bin)


class View(object):
    """
    Represents a :class:`View`, which consists of some :attr:`data` and a
    clustering algorithm, :attr:`clusterer`.

    Requires a data source, :attr:`data`, which is used to create the
    view (the data source can be a Python matrix (a list of lists), a
    NumPy 2D array, or a Pandas DataFrame) and a clustering method
    :attr:`clusterer`.

    Some examples follow (using a list of lists)::

        import pyrea

        data = [[1, 5, 3, 7],
                [4, 2, 9, 4],
                [8, 6, 1, 9],
                [7, 1, 8, 1]]

        v = pyrea.view(data, pyrea.cluster('ward'))

    Or by passing a Pandas DataFrame (``pandas.core.frame.DataFrame``)::

        import pyrea
        import pandas

        data = pandas.read_csv('iris.csv')

        v = pyrea.view(data, pyrea.cluster('ward'))

    Or (passing a numpy 2d array or matrix (``numpy.matrix`` or ``numpy.ndarray``))::

        import pyrea
        import numpy

        data = numpy.random.randint(0, 10, (4,4))

        v = pyrea.view(data, pyrea.cluster('ward'))


    .. seealso:: The :class:`Clusterer` class.

    :param data: The data from which to create your :class:`View`.
    :param clusterer: The clustering algorithm to use to cluster your
     :attr:`data`
    :ivar labels: Contains the calculated labels when the :attr:`clusterer`
     is run on the :attr:`data`.
    """
    def __init__(self, data, clusterer: List[Clusterer]) -> None:

        self.data = np.asarray(data)
        self.clusterer = clusterer
        self.labels = None

        if data.ndim != 2:
            raise Exception("Number of dimensions is not 2: you supplied a data structure with %s dimensions." % data.ndim)

    def execute(self) -> list:
        """
        Clusters the :attr:`data` using the :attr:`clusterer` specified at
        initialisation.
        """
        # TODO: If a list is passed, then we need to execute them all.
        self.labels = self.clusterer.execute(self.data)

        return self.labels


class Ward(Clusterer):
    """
    Implements the 'Ward' clustering algorithm.
    """
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Deprecated.")

    def execute(self, data):
        """
        Perform the clustering and return the results.
        """
        return AgglomerativeClustering().fit(data).labels_


class Complete(Clusterer):
    """
    Implements the 'complete' clustering algorithm.
    """
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Deprecated.")

    def execute(self, data):
        """
        Perform the clustering and return the results.
        """
        return AgglomerativeClustering(linkage='complete').fit(data).labels_


class Average(Clusterer):
    """
    Implements the 'average' clustering algorithm.
    """
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Deprecated.")

    def execute(self, data):
        """
        Perform the clustering and return the results.
        """
        return AgglomerativeClustering(linkage='average').fit(data).labels_


class Single(Clusterer):
    """
    Implements the 'single' clustering algorithm.
    """
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Deprecated.")

    def execute(self, data):
        """
        Perform the clustering and return the results.
        """
        return AgglomerativeClustering(linkage='single').fit(data).labels_


class Ensemble(object):
    """
    The Ensemble class encapsulates the views, fusion algorithm
    and clustering methods required to perform a multi-view clustering.

    :param views: The views that constitute the ensemble's multi-view data.
    :param fuser: The fusion algorithm to use.
    :param clusterers: The clustering algorithms to use on the fused matrix.
    """
    def __init__(self, views: List[View], fuser: Fusion):

        if isinstance(views, View):
            self.views = [views]
        elif isinstance(views, list):
            self.views = views

        self.fuser = fuser
        self.labels = []

    def execute(self):
        """
        Executes the ensemble, returning a :class:`View` object.

        The new :class:`View` can then be passed to subsequent ensembles.
        """

        # Execute each view's clustering algorithm on its data
        for v in self.views:
            self.labels.append(v.execute())

        # Fuse the clusterings to a single fused matrix
        fusion_matrix = self.fuser.execute(self.labels)

        return fusion_matrix
