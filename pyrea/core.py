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
import random
from array import array
from cmath import exp
from typing import List
import numpy as np
from sklearn.metrics import silhouette_score

# Genetic algorithm imports
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from .structure import Agreement, Clusterer, DBSCANPyrea, Disagreement
from .structure import Ensemble, Fusion, HierarchicalClusteringPyrea
from .structure import OPTICSPyrea, SpectralClusteringPyrea, View, Consensus

CLUSTER_METHODS = ['spectral', 'hierarchical', 'dbscan', 'optics']
FUSION_METHODS = ['agreement', 'consensus', 'disagreement']
LINKAGES = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward', 'ward2']

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
        else:
            kwargs['method'] = 'ward'

        if not kwargs['n_clusters']:
            raise TypeError("Error: n_clusters not set and is required for hierarchical clustering.")

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

def silhouette(labels: list):

    # Notes
    # We need to have our final model as an object where
    # you can pass data through it, in order to get predictions.
    # So once a fit has been performed, we need to be able to
    # get the labels.

    return silhouette_score

def parea_1(data: list, c_1_type='hierarchical', c_1_method='ward',
            c_2_type='hierarchical', c_2_method='complete',
            c_1_pre_type='hierarchical', c_1_pre_method='ward',
            c_2_pre_type='hierarchical', c_2_pre_method='complete',
            fusion_method='disagreement', k=2):
    """
    Implements the PAREA-1 algorithm.

    The function accepts a list of parameters for the Parea 1 algorithm.

    The default values are those described in the package's paper and README
    documentation regarding the PAREA-1 structure.

    .. seealso:: The :func:`~parea_2_genetic` function for a genetic algorithm
     optimised implementation of PAREA-2.

    :param data: A list of 3 NumPy matrices or 2D arrays.
    :param c_1_type: The clustering algorithm to use for the first ensemble.
    :param c_1_method: The clustering method to use for the first ensemble.
    :param c_2_type: The clustering algorithm to use for the second ensemble.
    :param c_2_method: The clustering method to use for the second ensemble.
    :param c_1_pre_type: The clustering algorithm to use for the first pre-computed ensemble.
    :param c_1_pre_method: The clustering method to use for the first pre-computed ensemble.
    :param c_2_pre_type: The clustering algorithm to use for the second pre-computed ensemble.
    :param c_2_pre_method: The clustering method to use for the second pre-computed ensemble.
    :param fusion_method: The fusion algorithm to use.
    :param k: The number of clusters to compute.
    """
    if len(data) != 3:
        raise ValueError("PAREA-1 requires exactly 3 data matrices.")

    # Clusterers:
    hc1 = clusterer(c_1_type, method=c_1_method, n_clusters=k)
    hc2 = clusterer(c_2_type, method=c_2_method, n_clusters=k)

    # Fusion algorithm:
    f = fuser(fusion_method)

    # Views for ensemble 1
    v1 = view(data[0], hc1)
    v2 = view(data[1], hc1)
    v3 = view(data[2], hc1)

    # Execute ensemble 1 and retrieve a new view, which is used later.
    hc1_pre = clusterer(c_1_pre_type, method=c_1_pre_method, n_clusters=k, precomputed=True)
    v_ensemble_1 = view(execute_ensemble([v1, v2, v3], f), hc1_pre)

    # Views for ensemble 2
    v4 = view(data[0], hc2)
    v5 = view(data[1], hc2)
    v6 = view(data[2], hc2)

    # Execute our second ensemble, and retreive a new view:
    hc2_pre = clusterer(c_2_pre_type, c_2_pre_method, n_clusters=k, precomputed=True)
    v_ensemble_2 = view(execute_ensemble([v4, v5, v6], f), hc2_pre)

    # Now we can execute a further ensemble, using the views generated from the
    # two previous ensemble methods:
    d_fuse  = execute_ensemble([v_ensemble_1, v_ensemble_2], f)

    # The returned distance matrix is now used as an input for the two clustering methods (hc1 and hc2)
    v1_fuse = view(d_fuse, hc1_pre)
    v2_fuse = view(d_fuse, hc2_pre)

    # and the cluster solutions are combined
    c = consensus([v1_fuse.execute(), v2_fuse.execute()])

    return c

def parea_1_genetic(data: list, max_k: int):
    """
    Genetic algorithm optimised implementation of Parea 1.
    """
    # Check if creator has been initialised
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define possible parameters
    # TODO: Replace these with the global variables
    cluster_methods = ['spectral', 'hierarchical', 'dbscan', 'optics']
    fusion_methods = ['agreement', 'consensus', 'disagreement']
    linkages = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward', 'ward2']

    # Cluster size range
    k_low, k_high = 2, max_k

    # Set up parameters for the genetic algorithm                       Index
    toolbox.register("c_1_type", random.choice, cluster_methods)        # 0
    toolbox.register("c_1_method", random.choice, linkages)             # 1
    toolbox.register("c_2_type", random.choice, cluster_methods)        # 2
    toolbox.register("c_2_method", random.choice, linkages)             # 3
    toolbox.register("c_1_pre_type", random.choice, cluster_methods)    # 4
    toolbox.register("c_1_pre_method", random.choice, linkages)         # 5
    toolbox.register("c_2_pre_type", random.choice, cluster_methods)    # 6
    toolbox.register("c_2_pre_method", random.choice, linkages)         # 7
    toolbox.register("fusion_method", random.choice, fusion_methods)    # 8
    toolbox.register("k", random.randint, k_low, k_high)                # 9

    # Chromosomes
    N_CYCLES = 1
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.c_1_type, toolbox.c_1_method, toolbox.c_2_type, toolbox.c_2_method, toolbox.c_1_pre_type, toolbox.c_1_pre_method, toolbox.c_2_pre_type, toolbox.c_2_pre_method, toolbox.fusion_method, toolbox.k), n=N_CYCLES)

    def mutate(individual):
        """ Mutate an individual. """
        # Choose a random parameter to mutate
        index = random.randint(0, len(individual) - 1)

        # Get the gene to be mutated
        gene = individual[index]

        # Mutate the parameter. Remember that:
        # 0, 2, 4, 6, are cluster methods
        # 1, 3, 5, 7, are linkages
        # 8 is the fusion method
        # 9 is the cluster size

        if gene in [0, 2, 4, 6]:
            individual[gene] = random.choice(cluster_methods)
        elif gene in [1, 3, 5, 7]:
            individual[gene] = random.choice(linkages)
        elif gene == 8:
            individual[gene] = random.choice(fusion_methods)
        elif gene == 9:
            individual[gene] = random.randint(k_low, k_high)

        return individual,

    def evaluate(individual):
        """ Evaluate an individual. """
        labels = parea_1(data, individual[0], individual[1], individual[2], individual[3], individual[4], individual[5], individual[6], individual[7], individual[8], individual[9])

        # Calculate the silhouette score
        s1 = silhouette_score(data[0], labels)
        s2 = silhouette_score(data[1], labels)
        s3 = silhouette_score(data[2], labels)

        return (s1 + s2 + s3) / 3,

    # Register the functions
    # TODO: Try other crossover methods and tournament sizes, etc.
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", evaluate)

    # Create the population
    population = toolbox.population(n=1000)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm
    x, y = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, stats=stats, halloffame=hall_of_fame, verbose=True)

    return hall_of_fame[0]

def cluster_silhoutte_score(data: list, type: str, method: str, k: int, precomputed: bool=False):

    # Cluster the data and get the labels
    c = clusterer(type, precomputed=precomputed, method=method, n_clusters=k)
    v = view(data, c)
    v.execute()

    return silhouette_score(data, v.labels, metric='precomputed' if precomputed else 'euclidean')

def parea_2(data: list, c_1_type='hierarchical', c_1_method='ward', c_1_k=2,
            c_2_type='hierarchical', c_2_method='complete', c_2_k=2,
            c_3_type='hierarchical', c_3_method='single', c_3_k=2,
            c_1_pre_type='hierarchical', c_1_pre_method='ward', c_1_pre_k=2,
            c_2_pre_type='hierarchical', c_2_pre_method='complete', c_2_pre_k=2,
            c_3_pre_type='hierarchical', c_3_pre_method='single', c_3_pre_k=2,
            fusion_method='disagreement', fitness=False):
    """
    Implements the PAREA-2 algorithm.

    :param views: A list of views to be used in the ensemble.
    :param c_1_type: The type of clustering algorithm to use for the first clustering step.
    :param c_1_method: The method of clustering algorithm to use for the first clustering step.
    :param c_2_type: The type of clustering algorithm to use for the second clustering step.
    :param c_2_method: The method of clustering algorithm to use for the second clustering step.
    :param c_3_type: The type of clustering algorithm to use for the third clustering step.
    :param c_3_method: The method of clustering algorithm to use for the third clustering step.
    :param c_1_pre_type: The type of clustering algorithm to use for the first clustering step (precomputed).
    :param c_1_pre_method: The method of clustering algorithm to use for the first clustering step (precomputed).
    :param c_2_pre_type: The type of clustering algorithm to use for the second clustering step (precomputed).
    :param c_2_pre_method: The method of clustering algorithm to use for the second clustering step (precomputed).
    :param c_3_pre_type: The type of clustering algorithm to use for the third clustering step (precomputed).
    :param c_3_pre_method: The method of clustering algorithm to use for the third clustering step (precomputed).
    :param fusion_method: The method of fusion to use.
    :param fitness: Whether to return the fitness or the labels.
    """

    if len(data) != 3:
        raise ValueError("PAREA-2 requires exactly 3 data matrices.")

    # Clustering algorithms
    c1 = clusterer(c_1_type, method=c_1_method, n_clusters=c_1_k)
    c2 = clusterer(c_2_type, method=c_2_method, n_clusters=c_2_k)
    c3 = clusterer(c_3_type, method=c_3_method, n_clusters=c_3_k)

    # Views
    v1 = view(data[0], c1)
    v2 = view(data[1], c2)
    v3 = view(data[2], c3)

    views = [v1, v2, v3]

    # Clustering algorithms (so it works with a precomputed distance matrix)
    c1_pre = clusterer(c_1_pre_type, method=c_1_pre_method, n_clusters=c_1_pre_k, precomputed=True)
    c2_pre = clusterer(c_2_pre_type, method=c_2_pre_method, n_clusters=c_2_pre_k, precomputed=True)
    c3_pre = clusterer(c_3_pre_type, method=c_3_pre_method, n_clusters=c_3_pre_k, precomputed=True)

    # Fusion algorithm
    f = fuser(fusion_method)

    # Create the ensemble and define new views based on the returned disagreement matrix v_res
    v_res  = execute_ensemble(views, f)
    v1_res = view(v_res, c1_pre)
    v2_res = view(v_res, c2_pre)
    v3_res = view(v_res, c3_pre)

    # Get the final cluster solution
    labels = consensus([v1_res.execute(), v2_res.execute(), v3_res.execute()])

    if fitness:
        return silhouette_score(v_res, labels, metric='precomputed')
    else:
        return labels

def parea_2_paper_genetic(data: list, max_k: int):
    """
    Parea 2 implementation as described in the paper.
    """

    pass

    k_list = []
    for d in data:
        best_k = 0
        best_s = 0
        for k in range(2, max_k + 1):
            for l in LINKAGES:
                s = cluster_silhoutte_score(d, 'hierarchical', l, k)
            if s > best_s:
                best_k = k

        k_list.append(best_k)

def parea_2_genetic(data: list, max_k: int):
    """
    Genetic algorithm optimised implementation of Parea 2.
    """
    # Create FitnesssMax and Individual classes/types.
    # Maybe place this outside of this function as it will be called
    # in the parea_1_genetic function also... and will be identical

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # Create the toolbox
    toolbox = base.Toolbox()

    # Cluster size range
    k_low, k_high = 2, max_k

    # Define possible parameters
    # TODO: Replace these with the global variables
    cluster_methods = ['spectral', 'hierarchical', 'dbscan', 'optics']
    fusion_methods = ['agreement', 'consensus', 'disagreement']
    linkages = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']

    # For testing
    # TODO: Remove this
    cluster_methods = ['hierarchical']
    fusion_methods = ['disagreement']
    # linkages = ['ward']

    # Set up our parameters for the genetic algorithm                   Index
    toolbox.register("c_1_type", random.choice, cluster_methods)        # 0
    toolbox.register("c_1_method", random.choice, linkages)             # 1
    toolbox.register("c_1_k", random.randint, k_low, k_high)            # 2
    toolbox.register("c_2_type", random.choice, cluster_methods)        # 3
    toolbox.register("c_2_method", random.choice, linkages)             # 4
    toolbox.register("c_2_k", random.randint, k_low, k_high)            # 5
    toolbox.register("c_3_type", random.choice, cluster_methods)        # 6
    toolbox.register("c_3_method", random.choice, linkages)             # 7
    toolbox.register("c_3_k", random.randint, k_low, k_high)            # 8
    toolbox.register("c_1_pre_type", random.choice, cluster_methods)    # 9
    toolbox.register("c_1_pre_method", random.choice, linkages)         # 10
    toolbox.register("c_1_pre_k", random.randint, k_low, k_high)        # 11
    toolbox.register("c_2_pre_type", random.choice, cluster_methods)    # 12
    toolbox.register("c_2_pre_method", random.choice, linkages)         # 13
    toolbox.register("c_2_pre_k", random.randint, k_low, k_high)        # 14
    toolbox.register("c_3_pre_type", random.choice, cluster_methods)    # 15
    toolbox.register("c_3_pre_method", random.choice, linkages)         # 16
    toolbox.register("c_3_pre_k", random.randint, k_low, k_high)        # 17
    toolbox.register("fusion_method", random.choice, fusion_methods)    # 18

    # How the chromosomes are created
    N_CYCLES = 1
    toolbox.register("individual", tools.initCycle, creator.Individual,
    (toolbox.c_1_type, toolbox.c_1_method, toolbox.c_1_k, toolbox.c_2_type,
    toolbox.c_2_method, toolbox.c_2_k,
    toolbox.c_3_type, toolbox.c_3_method, toolbox.c_3_k,
    toolbox.c_1_pre_type, toolbox.c_1_pre_method, toolbox.c_1_pre_k, toolbox.c_2_pre_type, toolbox.c_2_pre_method, toolbox.c_2_pre_k,
    toolbox.c_3_pre_type, toolbox.c_3_pre_method, toolbox.c_3_pre_k,
    toolbox.fusion_method), n=N_CYCLES)

    # How the population is created
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define how to mutate the chromosomes
    def mutate(inidividual):
        # Mutate one parameter randomly

        gene = random.randint(0, 13)

        if gene in [0, 3, 6, 9, 12, 15]:
            inidividual[gene] = random.choice(cluster_methods)
        elif gene in [1, 4, 7, 10, 13, 16]:
            inidividual[gene] = random.choice(linkages)
        elif gene in [2, 5, 8, 11, 14, 17]:
            inidividual[gene] = random.randint(k_low, k_high)
        elif gene == 18:
            inidividual[gene] = random.choice(fusion_methods)

        return inidividual,

    def evaluate(individual):

        sil = parea_2(data, individual[0], individual[1], individual[2], individual[3], individual[4], individual[5], individual[6], individual[7], individual[8], individual[9], individual[10], individual[11], individual[12], individual[13], individual[14], individual[15], individual[16], individual[17], individual[18], fitness=True)

        print("Function parea_2 called returned with sillhouette score: %s" % sil)

        return sil,

    # Register the functions
    # TODO: Try other crossover methods and tournament sizes, etc.
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", evaluate)

    # Create the population
    population = toolbox.population(n=100)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # TODO: Check what pop and log are
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, stats=stats, halloffame=hall_of_fame, verbose=True)

    return hall_of_fame[0]
