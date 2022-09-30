# Pyrea: Multi-view clustering with deep ensemble structures
# Copyright (C) 2022 Marcus D. Bloice, Bastian Pfeifer
#
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
from sklearn.cluster import AgglomerativeClustering
from typing import List

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

    def execute(self, data: list) -> list:
        """
        Execute the clustering algorithm with the given :attr:`data`.
        """
        pass


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
            res = np.matrix(res)
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
            res = np.matrix(res)
            labels = labels + res

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

        return(cl_cons)


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

        self.data = np.asmatrix(data)
        self.clusterer = clusterer
        self.labels = None

        # Numpy matrices can have max 2 dimensions, but can have 1 dimension.
        # If this needs to be checked revert to above below.
        #if data.ndim != 2:
        #    raise Exception("Number of dimensions is not 2: you supplied a data structure with %s dimensions." % data.ndim)

    def execute(self) -> list:
        """
        Clusters the :attr:`data` using the :attr:`clusterer` specified at
        initialisation.
        """
        # TODO: check the types here, do we expect a list of clusterers? Or one?
        self.labels = self.clusterer.execute(self.data)

        return self.labels


class Ward(Clusterer):
    """
    Implements the 'Ward' clustering algorithm.
    """
    def __init__(self) -> None:
        super().__init__()

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
    def __init__(self, views: List[View], fuser: Fusion, clusterers: List[Clusterer]):
        
        if isinstance(views, View):
            self.views = [views]
        elif isinstance(views, list):
            self.views = views

        if isinstance(clusterers, Clusterer):
            self.clusterers = [clusterers]
        elif isinstance(clusterers, list):
            self.clusterers = clusterers

        self.fuser = fuser
        self.labels = []
        self.computed_views = []

    def execute(self) -> View:
        """
        Executes the ensemble, returning a :class:`View` object.

        The new :class:`View` can then be passed to subsequent ensembles.
        """

        # Execute each view's clustering algorithm on its data
        for v in self.views:
            self.labels.append(v.execute())

        # Fuse the clusterings to a single fused matrix
        fusion_matrix = self.fuser.execute(self.labels)

        # Make new views with the fused matrix as data and the 
        # clutering algorithms that were passed.
        for i in range(len(self.clusterers)):
            self.computed_views.append(View(fusion_matrix, self.clusterers[i]))

        # Return the new view(s). Multiple views are returned if 
        # multiple clusters were specified.
        return self.computed_views
