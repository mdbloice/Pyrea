# Pyrea: Multi-view hierarchical clustering with flexible ensemble structures
# Copyright (C) 2022 Marcus D. Bloice, Bastian Pfeifer
#
# Licenced under the terms of the MIT license.

from cmath import exp
import numpy as np

CLUSTER_METHODS = ['ward', 'complete', 'single', 'average', 'random_method']


class ClusterMethod(object):
    """
    Top level abstract base class for clusters. Sub-types include
    :class:`DataCluster` and :class:`BinaryCluster`.
    """
    def __init__(self, clustermethod: str) -> None:

        if not isinstance(clustermethod, str):
            raise TypeError("Parameter 'clustermethod' must be of type string. Choices available are: %s."
                            % ("'" + "', '".join(CLUSTER_METHODS[:-1]) + "', or '" + CLUSTER_METHODS[-1] + "'"))

        if clustermethod not in CLUSTER_METHODS:
            raise TypeError("Parameter 'clustermethod' must be one of %s and you passed '%s'."
                            % ("'" + "', '".join(CLUSTER_METHODS[:-1]) + "', or '" + CLUSTER_METHODS[-1] + "'", clustermethod))


class View(object):
    """
    Represents a data view.

    :ivar data: Contains the data that initialised the object.
    :ivar binary_matrix: Contains the binary matrix if calculated.
    :ivar _is_calculated: Private member variable. Defines if the binary matrix
     has been calculated.
    """
    def __init__(self, data, clustermethod: ClusterMethod, header: list = None, name: str = None) -> None:
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

        # Numpy matrices are 2d-arrays, so checking dimensionality is not required.
        # Create Numpy matrix (revert to code above if n-dimensional arrays are required)
        data = np.asmatrix(data)

        if header is not None:
            if data.shape[1] != len(header):
                raise Exception("Number of columns in data parameter (%s) does not match number of headers provided (%s)"
                                % (data.shape[1], len(header)))

        self.data = data
        self._ncols = data.shape[1]
        self.name = name
        self.header = header
        self.binary_matrix = None
        self._is_calculated = False
        self._id = None  # Initially set to None, we give it an ID once added to workflow.

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


class Method(object):
    def __init__(self, method, k) -> None:

        self.method = method

        if self.k is None:
            self.k = 'auto'
        else:
            self.k = k


class Pipeline(object):
    def __init__(self, views: list = None) -> None:
        self.views = views

    def load_view() -> View:
        pass

def summary():
    """
    Prints a summary of the current pipeline, including any already calculated
    statistics.
    """
    title = "Summary Statistics"
    print(f" {title.title()} ".center(80, '*'))

    print("\n")
    print(f"Summary statistics to appear here".center(80))
    print("\n")

    print("*" * 80)


class Workflow(object):
    """
    Represents a workflow.

    :ivar elements: A list of tuples containing the elements of the workflow,
     for example, views, clusterers, and fusers.
    """
    def __init__(self, *args) -> None:


        self._num_elements = len(args)

        if args is not None:
            self.elements = args

            for i in range(len(args)):
                self.elements.append((i, args[i]))

        else:
            self.elements = []

    def add_view(self, view: View):
        self._num_elements += 1
        view._id = self._num_elements

        self.elements.append(view)

    def add_clusterer(self, cluster: ClusterMethod):
        self.elements.append(cluster)


class FusedMatrix(View):
    pass