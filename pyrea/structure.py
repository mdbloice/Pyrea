# Pyrea: Multi-view hierarchical clustering with flexible ensemble structures
# Copyright (C) 2022 Marcus D. Bloice, Bastian Pfeifer
#
# Licenced under the terms of the MIT license.
#
# structure.py
# Contains the classes required for the structuring of ensembles, for example
# Views, Ensembles, Clusterers, and so on.

from re import A
import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from .fusion import disagreement

class Fusion(object):
    def __init__(self) -> None:
        pass
    
    def execute():
        pass

class Parea(Fusion):
    def __init__(self) -> None:
        super().__init__()
    
    def execute():
        pass


class Disagreement(Fusion):
    def __init__(self, views: list) -> None:
        super().__init__()
        
        self.labels = None
        self.views = []

        # Check all views are of the same length
        if not all(len(views[0]) == len(_) for _ in views):
            raise TypeError("The size of thew views must be equal.")
        else:
            self.views = views

    def execute(self):

        n  = len(self.views[0])
        labels  = np.zeros((n, n), dtype=int)
    
        for view_index in range(0, len(self.views)):

            l = self.views[view_index]
            res = [[int(x != y) for y in l] for x in l]
            res = np.matrix(res)
            mat = mat + res

            return(mat)

class Clusterer(object):
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        return str(self.labels)

    def execute():
        pass


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
        self._is_calculated = False
        self._id = None  # Initially set to None, we give it an ID once added to workflow.
        self._cluster = None

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

    def get_binary_matrix(self) -> np.ndarray:
        """
        Returns the binary matrix of your view. If no binary matrix has been
        calculated a NameError is thrown.
        """
        if not self.binary_matrix:
            raise NameError('A binary matrix has not been calculated and does not exist')
        else:
            return self.binary_matrix

    def execute(self):
        self._cluster = self.clustermethod.execute(self.data)


class Ward(Clusterer):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.labels = None

    def execute(self):
        self.labels = AgglomerativeClustering().fit(self.data).labels_
        return self


class KMeans(Clusterer):
    def __init__(self) -> None:
        super().__init__()

    def execute(self, view: View):
        print("Executing clustering algorithm")
        time.sleep(3)


class Ensemble(object):
    def __init__(self, views: list, fuser: list) -> None:

        # If we see 1 clusterer, we use it for all views. Change this.
        #if len(views) is not len(views):
        #    raise Exception("Number of elements (views or ensembles) (%s) does not match number of clusterers (%s)." % (len(views), len(clusterers)))

        self.views = views
        self.fuser = fuser
        self._fusion_matrix = None

    def execute(self):

        pass