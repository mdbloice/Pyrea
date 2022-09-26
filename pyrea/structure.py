# Pyrea: Multi-view clustering with deep ensemble structures
# Copyright (C) 2022 Marcus D. Bloice, Bastian Pfeifer
#
#
# Licenced under the terms of the MIT license.
#
# structure.py
# Contains the classes required for the structuring of ensembles, for
# example Views, Ensembles, Clusterers, and so on.

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .fusion import disagreement


class Clusterer(object):
    """
    Abstract Base Class for all clustering algorithms.

    All clustering methods must implement the base class's methods
    in order for it to be used as a clusterer within Pyrea.
    """
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return str(self.labels)

    def execute():
        """
        Execute the clustering algorithm. For internal use only.
        """
        pass


class Fusion(object):
    def __init__(self) -> None:
        pass

    def execute():
        pass


class Parea(Fusion):
    def __init__(self) -> None:
        super().__init__()
        raise Exception("Not yet implemented.")

    def execute():
        pass


class Disagreement(Fusion):
    def __init__(self) -> None:
        super().__init__()

        # Check all views are of the same length
        #if not all(len(views[0]) == len(_) for _ in views):
        #    raise TypeError("The size of thew views must be equal.")
        #else:
        #    self.views = views

    def execute(self, views: list) -> list:

        n  = len(views[0])
        labels = np.zeros((n, n), dtype=int)

        for i in range(0, len(views)):
            l = views[i]
            res = [[int(x != y) for y in l] for x in l]
            res = np.matrix(res)
            labels = labels + res

        return labels


class Agreement(Fusion):
    def __init__(self) -> None:
        super().__init__()
        pass

    def execute(views: list):

        n_samp  = len(views[0])

        labels  = np.zeros((n_samp, n_samp), dtype=int)

        for i in range(0, len(views)):

            l = views[i]
            res = [[int(x == y) for y in l] for x in l]
            res = np.matrix(res)
            labels = labels + res

        return labels

class View(object):
    """
    Represents a data view.

    :ivar data: Contains the data that initialised the object.
    :ivar binary_matrix: Contains the binary matrix if calculated.
    :ivar _is_calculated: Private member variable. Defines if the binary matrix
     has been calculated.
    """
    def __init__(self, data, clusterer: Clusterer, header: list = None, name: str = None) -> None:
        """
        Initilisation: A data source, :attr:`data`, which is used to create the
        view. The data source can be a Python matrix (a list of lists), a
        NumPy 2D array, or a Pandas DataFrame.

        .. seealso:: The :class:`Cluster` class.

        For example::

            import pyrea

            data = [[1, 5, 3, 7],
                    [4, 2, 9, 4],
                    [8, 6, 1, 9],
                    [7, 1, 8, 1]]

            v = pyrea.View(data)

        Or by passing a Pandas DataFrame (``pandas.core.frame.DataFrame``)::

            import pyrea
            import pandas

            data = pandas.read_csv('iris.csv')

            v = pyrea.View(data)

        Or (passing a numpy 2d array or matrix (``numpy.matrix`` or ``numpy.ndarray``))::

            import pyrea
            import numpy

            data = numpy.random.randint(0, 10, (4,4))

            v = pyrea.View(data)

        :param data: The data from which to create your view.
        :type data: Python matrix (list of lists), NumPy Array, Pandas DataFrame
        """
        # We shall attempt to create a NumPy matrix from the data passed:
        #data = np.asarray(data)
        #if data.ndim != 2:
        #    raise Exception("Number of dimensions is not 2: you supplied a data structure with %s dimensions." % data.ndim)

        # Numpy matrices can have max 2 dimensions, but can have 1 dimension. If this needs to be checked revert to above code.
        data = np.asmatrix(data)

        if header is not None:
            if data.shape[1] != len(header):
                raise Exception("Number of columns in data parameter (%s) does not match number of headers provided (%s)"
                                % (data.shape[1], len(header)))

        self.data = data
        self.clusterer = clusterer
        self._ncols = data.shape[1]
        self.name = name
        self.header = header
        self._id = None  # Initially set to None, we give it an ID once added to workflow.
        self.labels = None

    def summary(self) -> None:
        """
        Print some summary statistics regarding the current view.

        For example::

            import pyrea
            import pandas

            data = pandas.read_csv('iris.csv')
            v = pyrea.View(data)

            v.summary()

        """
        print('Summary statistics.')
        return None

    def execute(self):

        print("Running view.execute()...")
        self.labels = self.clusterer.execute(self.data)

        return self.labels


class Ward(Clusterer):
    def __init__(self) -> None:
        super().__init__()

    def execute(self, data):

        return AgglomerativeClustering(linkage='ward').fit(data).labels_


class Complete(Clusterer):
    def __init__(self) -> None:
        super().__init__()

    def execute(self, data):
        return AgglomerativeClustering(linkage='average').fit(data).labels_


class Ensemble(object):
    def __init__(self, elements: list, fuser: Fusion, clusterers: list):

        self.elements = elements
        self.fuser = fuser
        self.clusterers = clusterers
        self.clusters = []
        self.labels = []
        self.fusion_matrix = None

    def execute(self):

        # Check if this ensemble contains further ensembles, or views.
        # If we contain ensembles, we need to iterate over these and allow
        # them to execute first. If they themselves contain ensembles the same
        # will happen there.

        print ("Running ensemble.execute()...")
        for e in self.elements:

            self.clusters.append(e.execute())

        self.fusion_matrix = self.fuser.execute(self.clusters)

        for c in self.clusterers:
            self.labels.append(c.execute(self.fusion_matrix))

        return self.labels[0]
