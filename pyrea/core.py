# Pyrea: Multi-view hierarchical clustering with flexible ensemble structures
# Copyright (C) 2022 Marcus D. Bloice, Bastian Pfeifer
#
# Licenced under the terms of the MIT license.

from cmath import exp

from .structure import View
from .structure import Ward

CLUSTER_METHODS = ['ward', 'complete', 'single', 'average', 'random']

def new_clusterer(clusterer: str, data):
    if not isinstance(clusterer, str):
        raise TypeError("Parameter 'clusterer' must be of type string. Choices available are: %s."
                        % ("'" + "', '".join(CLUSTER_METHODS[:-1]) + "', or '" + CLUSTER_METHODS[-1] + "'"))

    if clusterer not in CLUSTER_METHODS:
        raise TypeError("Parameter 'clusterer' must be one of %s and you passed '%s'."
                        % ("'" + "', '".join(CLUSTER_METHODS[:-1]) + "', or '" + CLUSTER_METHODS[-1] + "'", clusterer))

    if clusterer is 'ward':
        return Ward(data=data)
    else:
        return None

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

    print(f" End Summary ".center(80, "*"))


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

    def add_clusterer(self, cluster: object):
        self.elements.append(cluster)
