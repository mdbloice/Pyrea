# Pyrea: Multi-view hierarchical clustering with flexible ensemble structures
# Copyright (C) 2022 Marcus D. Bloice, Bastian Pfeifer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses/

import numpy as np

class View(object):
    """
    Represents a data view.

    :ivar data: Contains the data that initialised the object.
    :ivar binary_matrix: Contains the binary matrix if calculated.
    :ivar _is_calculated: Private member variable. Defines if the binary matrix
     has been calculated.
    """
    def __init__(self, data) -> None:
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

        Or (passing a Pandas DataFrame)::

            import pyrea
            import pandas

            data = pandas.read_csv('iris.csv')

            v = pyrea.View(data)

        Or (passing a NumPy array (``ndarray``))::

            import pyrea
            import numpy

            data = numpy.random.randint(0, 10, (4,4))

            v = pyrea.View(data)

        :param data: The data from which to create your view.
        :type data: Python matrix (list of lists), NumPy Array, Pandas DataFrame
        """
        self.data = data
        self.binary_matrix = None
        self._is_calculated = False

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


class Cluster(object):
    """
    Top level abstract base class for clusters. Sub-types include
    :class:`DataCluster` and :class:`BinaryCluster`.
    """

class DataCluster(Cluster):
    """
    A :class:`Cluster` containing non-binary fused data.
    """

class BinaryCluster(Cluster):
    """
    A :class:`Cluster` containing binary fused cluster data.
    """

class ClusterFuser(object):
    """
    The ClusterFuser fuses clusters that have been loaded.
    """
    def __init__(self) -> None:
        pass


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
