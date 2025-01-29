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
from typing import List, Union
import numpy as np
import warnings
from sklearn.metrics import silhouette_score
from operator import itemgetter

# Genetic algorithm imports
from deap import base, creator, tools, algorithms

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

def parea_2_mv(data: list, clusterers: list, methods: list, k_s: list, precomputed_clusterers: list, precomputed_methods: list, precomputed_k_s: list, fusion_method='disagreement', fitness=False, k_final=None):

    # Sanity check
    if not len(data) == len(clusterers) == len(methods) == len(k_s):
        raise ValueError("The number of views, clusterers, methods and k_s must be equal.")

    if not len(precomputed_clusterers) == len(precomputed_methods) == len(precomputed_k_s):
        raise ValueError("The number of precomputed clusterers, precomputed methods and precomputed k_s must be equal.")

    # create a structure based on the number of views
    n_views = len(data)

    # Clustering algorithms
    clustering_algorithms = [None] * n_views
    for i in range(n_views):
        clustering_algorithms[i] = clusterer(clusterers[i], method=methods[i], n_clusters=k_s[i])

    # Pre-computed clustering algorithms
    precomputed_clustering_algorithms = [None] * len(precomputed_clusterers)
    for i in range(n_views):
        precomputed_clustering_algorithms[i] = clusterer(precomputed_clusterers[i], method=precomputed_methods[i], n_clusters=precomputed_k_s[i], precomputed=True)

    # Create the views
    # TODO: Check if n_views is correct here...
    views = [None] * n_views
    for i in range(n_views):
        views[i] = view(data[i], clustering_algorithms[i])

    # Create fusion algorithm
    f = fuser(fusion_method)

    # Create ensemble
    v_res = execute_ensemble(views, f)

    # Always set to the 0th precomputed clustering algorithm and method...
    # TODO: Can/should this be changed?
    if k_final:
        c_final = clusterer(precomputed_clusterers[0], method=precomputed_methods[0], n_clusters=k_final, precomputed=True)
        v_res_final = view(v_res, c_final)
        labels = v_res_final.execute()

        # Remove the last dimension of the labels, could also use np.squeeze, but requires 1.7 or higher
        labels = labels[:,0]
    else:
        v_res_array = [None] * len(precomputed_clusterers)

        for i in range(len(precomputed_clusterers)):
            v_res_array[i] = view(v_res, precomputed_clustering_algorithms[i])

        # Get the final cluster solution
        labels = consensus([v_res.execute() for v_res in v_res_array])

    if fitness:
        return silhouette_score(v_res, labels, metric='precomputed')
    else:
        return labels

def parea_2_mv_genetic(data: list, k_min: int, k_max: int, k_final: Union[int, None] = None, n_population=100, n_generations=10):

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # Create the toolbox
    toolbox = base.Toolbox()

    cluster_methods = ['hierarchical']
    fusion_methods = ['disagreement']
    linkages = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward', 'ward2']

    # Attribute generator
    n = len(data)
    names = []
    c_indices = []

    # Create our clusterers. We do not want 0-based names so i+1 is used to generate the name strings.
    for i in range(n):
        toolbox.register(f"c_{i+1}_type", random.choice, cluster_methods)
        names.append(f"c_{i+1}_type")

    # Create our methods
    for i in range(n):
        toolbox.register(f"c_{i+1}_method" % i+1, random.choice, linkages)
        names.append(f"c_{i+1}_method")

    # Create our k_s
    for i in range(n):
        toolbox.register(f"c_{i+1}_k", random.randint, k_min, k_max)
        names.append(f"c_{i+1}_k")

    # Create our precomputed clusterers
    for i in range(n):
        toolbox.register(f"c_{i+1}_pre_type", random.choice, cluster_methods)
        names.append(f"c_{i+1}_pre_type")

    # Create our precomputed methods
    for i in range(n):
        toolbox.register(f"c_{i+1}_pre_method", random.choice, linkages)
        names.append(f"c_{i+1}_pre_method")

    # Create our precomputed k_s
    for i in range(n):
        toolbox.register(f"c_{i+1}_pre_k", random.randint, k_min, k_max)
        names.append(f"c_{i+1}_pre_k")

    toolbox.register("fusion_method", random.choice, fusion_methods)
    names.append("fusion_method")

    to_pass = tuple([getattr(toolbox, name) for name in names])

    # Chromosome structure initialiser
    N_CYCLES = 1
    toolbox.register("individual", tools.initCycle, creator.Individual, to_pass, N_CYCLES)

    # Population initialiser
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Index builder
    clusterers_index        = [*range(0, len(data))]
    methods_index           = [*range(len(data)*1, len(data)*2)]
    k_s_index               = [*range(len(data)*2, len(data)*3)]
    pre_clusterers_index    = [*range(len(data)*3, len(data)*4)]
    pre_methods_index       = [*range(len(data)*4, len(data)*5)]
    pre_k_s_index           = [*range(len(data)*5, len(data)*6)]
    f_index                 = len(names)-1

    print(names)

    # Mutation function
    def mutate(individual):
        """
        Mutates an individual by randomly changing a single parameter.
        """
        index = random.randint(0, len(individual) - 1)
        gene = individual[index]

        # TODO: replace with indices above.
        if gene in [*range(0, len(data))]:
            individual[gene] = random.choice(cluster_methods)
        elif gene in [*range(len(data)*1, len(data)*2)]:
            individual[gene] = random.choice(linkages)
        elif gene in [*range(len(data)*2, len(data)*3)]:
            individual[gene] = random.randint(k_min, k_max)
        elif gene in [*range(len(data)*3, len(data)*4)]:
            individual[gene] = random.choice(cluster_methods)
        elif gene in [*range(len(data)*4, len(data)*5)]:
            individual[gene] = random.choice(linkages)
        elif gene in [*range(len(data)*5, len(data)*6)]:
            individual[gene] = random.randint(k_min, k_max)
        elif gene == len(names)-1:
            individual[gene] = random.choice(fusion_methods)

        return individual,

    def evaluate(individual):

        # sklearn uses np.matrix which throws a warning as it is deprecated
        # Use this to silence the warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sil = parea_2_mv(data,
                            clusterers=itemgetter(*clusterers_index)(individual),
                            methods=itemgetter(*methods_index)(individual),
                            k_s=itemgetter(*k_s_index)(individual),
                            precomputed_clusterers=itemgetter(*pre_clusterers_index)(individual),
                            precomputed_methods=itemgetter(*pre_methods_index)(individual),
                            precomputed_k_s=itemgetter(*pre_k_s_index)(individual),
                            fusion_method=individual[f_index],
                            fitness=True, k_final=k_final)

        print("Silhouette score: %s" % sil)

        return sil,

    # Register the functions
    # TODO: Try other crossover methods and tournament sizes, etc.
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", evaluate)

    # Create the population
    population = toolbox.population(n=n_population)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # TODO: Work out if pop and log should also be returned
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations, stats=stats, halloffame=hall_of_fame, verbose=True)

    print(f"\nSummary:\n{log}")

    # We need to convert the optimal parameters to a format understood by the parea_2_mv function.
    return convert_to_parameters(len(data), hall_of_fame[0])

def parea_1(data: list, c_1_type='hierarchical', c_1_method='ward', c_1_k=2,
            c_2_type='hierarchical', c_2_method='complete', c_2_k=2,
            c_1_pre_type='hierarchical', c_1_pre_method='ward', c_1_pre_k=2,
            c_2_pre_type='hierarchical', c_2_pre_method='complete', c_2_pre_k=2,
            fusion_method='disagreement', fitness=False, k_final=None):
    """
    Implements the PAREA-1 algorithm.

    The function accepts a list of parameters for the Parea 1 algorithm.

    The default values are those described in the package's paper and README
    documentation regarding the PAREA-1 structure.

    .. seealso:: The :func:`~parea_2_genetic` function for a genetic algorithm
     optimised implementation of PAREA-2.

    :param data: A list of Numpy matrices or 2D arrays.
    :param c_1_type: The clustering algorithm to use for the first ensemble.
    :param c_1_method: The clustering method to use for the first ensemble.
    :param c_2_type: The clustering algorithm to use for the second ensemble.
    :param c_2_method: The clustering method to use for the second ensemble.
    :param c_1_pre_type: The clustering algorithm to use for the first pre-computed ensemble.
    :param c_1_pre_method: The clustering method to use for the first pre-computed ensemble.
    :param c_2_pre_type: The clustering algorithm to use for the second pre-computed ensemble.
    :param c_2_pre_method: The clustering method to use for the second pre-computed ensemble.
    :param fusion_method: The fusion algorithm to use.
    """

    # Clustering algorithms
    c1 = clusterer(c_1_type, method=c_1_method, n_clusters=c_1_k)
    c2 = clusterer(c_2_type, method=c_2_method, n_clusters=c_2_k)

    # Clustering algorithms (so it works with a precomputed distance matrix)
    c1_pre = clusterer(c_1_pre_type, method=c_1_pre_method, n_clusters=c_1_pre_k, precomputed=True)
    c2_pre = clusterer(c_2_pre_type, method=c_2_pre_method, n_clusters=c_2_pre_k, precomputed=True)

    # Views for ensemble 1
    views1 = []
    for v in data:
        views1.append(view(v, c1))

    # Fusion algorithm:
    f = fuser(fusion_method)

    # Execute our first ensemble, and retreive a new view:
    v_ensemble_1 = view(execute_ensemble(views1, f), c1_pre)

    # Views for ensemble 2
    views2 = []
    for v in data:
        views2.append(view(v, c2))

    # Execute our second ensemble, and retreive a new view:
    v_ensemble_2 = view(execute_ensemble(views2, f), c2_pre)

    # Now we can execute a further ensemble, using the views generated from the
    # two previous ensemble methods:
    v_res  = execute_ensemble([v_ensemble_1, v_ensemble_2], f)

    # TODO: allow for type and method to be passed in as parameters
    if k_final:
        c_final = clusterer(c_1_pre_type, method=c_1_pre_method, n_clusters=k_final, precomputed=True)
        v_res_final = view(v_res, c_final)
        labels = v_res_final.execute()

        labels = labels[:,0]
    else:
        # The returned distance matrix is now used as an input for the two clustering methods (hc1 and hc2)
        v1_res = view(v_res, c1_pre)
        v2_res = view(v_res, c2_pre)

        # and the cluster solutions are combined
        labels = consensus([v1_res.execute(), v2_res.execute()])

    if fitness:
        return silhouette_score(v_res, labels, metric='precomputed')
    else:
        return labels

def parea_1_spectral(data: list,
            c_1_type='spectral', c_1_n_neighbors=10, c_1_k=2,
            c_2_type='spectral', c_2_n_neighbors=10, c_2_k=2,
            c_1_pre_type='spectral', c_1_pre_n_neighbors=10, c_1_pre_k=2,
            c_2_pre_type='spectral', c_2_pre_n_neighbors=10, c_2_pre_k=2,
            fusion_method='agreement', fitness=False, k_final=None):

    # NOTE: All code below here is almost identical to parea_1, except for the
    # clusterer. Therefore, it would make sense to refactor this code to
    # remove the duplication.
    c1 = clusterer(c_1_type, n_neighbors=c_1_n_neighbors, n_clusters=c_1_k)
    c2 = clusterer(c_2_type, n_neighbors=c_2_n_neighbors, n_clusters=c_2_k)

    c1_pre = clusterer(c_1_pre_type, n_neighbors=c_1_pre_n_neighbors, n_clusters=c_1_pre_k, precomputed=True)
    c2_pre = clusterer(c_2_pre_type, n_neighbors=c_2_pre_n_neighbors, n_clusters=c_2_pre_k, precomputed=True)

    # Views for ensemble 1
    views1 = []
    for v in data:
        views1.append(view(v, c1))

    # Fusion algorithm:
    f = fuser(fusion_method)

    # Execute our first ensemble, and retreive a new view:
    v_ensemble_1 = view(execute_ensemble(views1, f), c1_pre)

    # Views for ensemble 2
    views2 = []
    for v in data:
        views2.append(view(v, c2))

    # Execute our second ensemble, and retreive a new view:
    v_ensemble_2 = view(execute_ensemble(views2, f), c2_pre)

    # Now we can execute a further ensemble, using the views generated from the
    # two previous ensemble methods:
    v_res  = execute_ensemble([v_ensemble_1, v_ensemble_2], f)

    if k_final:
        c_final = clusterer(c_1_pre_type, n_neighbors=c_1_pre_n_neighbors, n_clusters=k_final, precomputed=True)
        v_res_final = view(v_res, c_final)
        # Labels are returned differently in spectral clustering, so we need to
        # change how these are returned (compared to hierarchical clustering)
        labels = v_res_final.execute()

    else:
        # The returned distance matrix is now used as an input for the two clustering methods (hc1 and hc2)
        v1_res = view(v_res, c1_pre)
        v2_res = view(v_res, c2_pre)

        # and the cluster solutions are combined
        labels = consensus([v1_res.execute(), v2_res.execute()])

    if fitness:
        v_res_disagreement = 1 - (v_res/v_res.max())
        np.fill_diagonal(v_res_disagreement, 0)  # DONE IN PLACE!
        return silhouette_score(v_res_disagreement, labels, metric='precomputed')
    else:
        return labels

def parea_1_genetic_spectral(data: list, k_min: int, k_max: int, n_neighbors_min: int, n_neighbors_max: int, k_final: Union[int, None] = None, n_population=100, n_generations=10):
    """
    Genetic algorithm optimised implementation of Parea 1 with spectral clustering.
    """

    # Check if creator has been initialised
    # TODO: Check if this is required.
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # TODO: Check that we can use more fusion methods with Spectral
    cluster_methods = ['spectral']
    fusion_methods = ['agreement']  # MUST ALWAYS BE AGREEMENT FOR SPECTRAL

    # Set up parameters for the genetic algorithm                                               Index
    toolbox.register("c_1_type", random.choice, cluster_methods)                                # 0
    toolbox.register("c_1_n_neighbors", random.randint, n_neighbors_min, n_neighbors_max)       # 1
    toolbox.register("c_1_k", random.randint, k_min, k_max)                                     # 2
    toolbox.register("c_2_type", random.choice, cluster_methods)                                # 3
    toolbox.register("c_2_n_neighbors", random.randint, n_neighbors_min, n_neighbors_max)       # 4
    toolbox.register("c_2_k", random.randint, k_min, k_max)                                     # 5
    toolbox.register("c_1_pre_type", random.choice, cluster_methods)                            # 6
    toolbox.register("c_1_pre_n_neighbors", random.randint, n_neighbors_min, n_neighbors_max)   # 7
    toolbox.register("c_1_pre_k", random.randint, k_min, k_max)                                 # 8
    toolbox.register("c_2_pre_type", random.choice, cluster_methods)                            # 9
    toolbox.register("c_2_pre_n_neighbors", random.randint, n_neighbors_min, n_neighbors_max)   # 10
    toolbox.register("c_2_pre_k", random.randint, k_min, k_max)                                 # 11
    toolbox.register("fusion_method", random.choice, fusion_methods)                            # 12

    # TODO: Let user define this
    N_CYCLES = 1
    toolbox.register("individual", tools.initCycle, creator.Individual,
        (toolbox.c_1_type, toolbox.c_1_n_neighbors, toolbox.c_1_k,
        toolbox.c_2_type, toolbox.c_2_n_neighbors, toolbox.c_2_k,
        toolbox.c_1_pre_type, toolbox.c_1_pre_n_neighbors, toolbox.c_1_pre_k,
        toolbox.c_2_pre_type, toolbox.c_2_pre_n_neighbors, toolbox.c_2_pre_k,
        toolbox.fusion_method), n=N_CYCLES)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def mutate(individual):
        index = random.randint(0, len(individual) - 1)
        gene = individual[index]

        # Refer to indices above using toolbox.register
        if gene in [0, 3, 6, 9]:
            individual[gene] = random.choice(cluster_methods)
        elif gene in [1, 4, 7, 10]:
            individual[gene] = random.randint(n_neighbors_min, n_neighbors_max)
        elif gene in [2, 5, 8, 11]:
            individual[gene] = random.randint(k_min, k_max)
        elif gene == 12:
            individual[gene] = random.choice(fusion_methods)

        return individual,

    def evaluate(individual):
        """ Evaluate an individual. """

        # sklearn uses np.matrix which throws a warning as it is deprecated
        # Use this to silence the warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sil = parea_1_spectral(data,
                            individual[0],
                            individual[1],
                            individual[2],
                            individual[3],
                            individual[4],
                            individual[5],
                            individual[6],
                            individual[7],
                            individual[8],
                            individual[9],
                            individual[10],
                            individual[11],
                            individual[12],
                            fitness=True, k_final=k_final)

        print("Silhouette score: %s" % sil)

        return sil,

    # Register the functions
    # TODO: Try other crossover methods and tournament sizes, etc.
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", evaluate)

    # Create the population
    population = toolbox.population(n=n_population)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations, stats=stats, halloffame=hall_of_fame, verbose=True)

    print(f"\nSummary:\n{log}")

    return hall_of_fame[0]

def parea_1_genetic(data: list, k_min: int, k_max: int, k_final: Union[int, None] = None, n_population=100, n_generations=10):
    """
    Genetic algorithm optimised implementation of Parea 1.

    Use family='spectral' for spectral clustering. Not yet implemented.
    """

    # Check if creator has been initialised
    # TODO: Check if this is required.
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

    # Overwrite the cluster and fusion methods temporarily
    cluster_methods = ['hierarchical']
    fusion_methods = ['disagreement']

    # Set up parameters for the genetic algorithm                       Index
    toolbox.register("c_1_type", random.choice, cluster_methods)        # 0
    toolbox.register("c_1_method", random.choice, linkages)             # 1
    toolbox.register("c_1_k", random.randint, k_min, k_max)             # 2
    toolbox.register("c_2_type", random.choice, cluster_methods)        # 3
    toolbox.register("c_2_method", random.choice, linkages)             # 4
    toolbox.register("c_2_k", random.randint, k_min, k_max)             # 5
    toolbox.register("c_1_pre_type", random.choice, cluster_methods)    # 6
    toolbox.register("c_1_pre_method", random.choice, linkages)         # 7
    toolbox.register("c_1_pre_k", random.randint, k_min, k_max)         # 8
    toolbox.register("c_2_pre_type", random.choice, cluster_methods)    # 9
    toolbox.register("c_2_pre_method", random.choice, linkages)         # 10
    toolbox.register("c_2_pre_k", random.randint, k_min, k_max)         # 11
    toolbox.register("fusion_method", random.choice, fusion_methods)    # 12

    # Chromosomes
    # TODO: Add user parameter for N_CYCLES
    N_CYCLES = 1
    toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.c_1_type, toolbox.c_1_method, toolbox.c_1_k,
                        toolbox.c_2_type, toolbox.c_2_method, toolbox.c_2_k,
                        toolbox.c_1_pre_type, toolbox.c_1_pre_method, toolbox.c_1_pre_k,
                        toolbox.c_2_pre_type, toolbox.c_2_pre_method, toolbox.c_2_pre_k,
                        toolbox.fusion_method), n=N_CYCLES)

    # How the population is created
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def mutate(individual):
        """ Mutate an individual. """
        # Choose a random parameter to mutate, and select the gene
        index = random.randint(0, len(individual) - 1)
        gene = individual[index]

        # Mutate the parameter. Remember that:
        # 0, 3, 6, 9, are cluster methods
        # 1, 4, 7, 10, are linkages
        # 2, 5, 8, 11, are cluster sizes
        # 12 is the fusion method

        if gene in [0, 3, 6, 9]:
            individual[gene] = random.choice(cluster_methods)
        elif gene in [1, 4, 7, 10]:
            individual[gene] = random.choice(linkages)
        elif gene in [2, 5, 8, 11]:
            individual[gene] = random.randint(k_min, k_max)
        elif gene == 12:
            individual[gene] = random.choice(fusion_methods)

        return individual,

    def evaluate(individual):
        """ Evaluate an individual. """

        # sklearn uses np.matrix which throws a warning as it is deprecated
        # Use this to silence the warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sil = parea_1(data,
                            individual[0],
                            individual[1],
                            individual[2],
                            individual[3],
                            individual[4],
                            individual[5],
                            individual[6],
                            individual[7],
                            individual[8],
                            individual[9],
                            individual[10],
                            individual[11],
                            individual[12],
                            fitness=True, k_final=k_final)

        print("Silhouette score: %s" % sil)

        return sil,

    # Register the functions
    # TODO: Try other crossover methods and tournament sizes, etc.
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("evaluate", evaluate)

    # Create the population
    population = toolbox.population(n=n_population)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_generations, stats=stats, halloffame=hall_of_fame, verbose=True)

    print(f"\nSummary:\n{log}")

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
            fusion_method='disagreement', fitness=False, k_final=None):
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

    # Clustering algorithms (so it works with a precomputed distance matrix)
    c1_pre = clusterer(c_1_pre_type, method=c_1_pre_method, n_clusters=c_1_pre_k, precomputed=True)
    c2_pre = clusterer(c_2_pre_type, method=c_2_pre_method, n_clusters=c_2_pre_k, precomputed=True)
    c3_pre = clusterer(c_3_pre_type, method=c_3_pre_method, n_clusters=c_3_pre_k, precomputed=True)

    # Views
    v1 = view(data[0], c1)
    v2 = view(data[1], c2)
    v3 = view(data[2], c3)

    views = [v1, v2, v3]

    # Fusion algorithm
    f = fuser(fusion_method)

    # Create the ensemble and define new views based on the returned disagreement matrix v_res
    v_res  = execute_ensemble(views, f)

    # TODO: allow for type and method to be passed in as parameters
    if k_final:
        c_final = clusterer(c_1_pre_type, method=c_1_pre_method, n_clusters=k_final, precomputed=True)
        v_res_final = view(v_res, c_final)
        labels = v_res_final.execute()

        # Remove the last dimension of the labels, could also use np.squeeze, but requires 1.7 or higher
        labels = labels[:,0]
    else:
        v1_res = view(v_res, c1_pre)
        v2_res = view(v_res, c2_pre)
        v3_res = view(v_res, c3_pre)

        # Get the final cluster solution
        labels = consensus([v1_res.execute(), v2_res.execute(), v3_res.execute()])

    if fitness:
        return silhouette_score(v_res, labels, metric='precomputed')
    else:
        return labels

def parea_2_genetic(data: list, k_min: int, k_max: int, k_final: Union[int, None] = None):
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

    # Define possible parameters
    # TODO: Replace these with the global variables
    cluster_methods = ['spectral', 'hierarchical', 'dbscan', 'optics']
    fusion_methods = ['agreement', 'consensus', 'disagreement']
    linkages = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward', 'ward2']

    # Overwrite the cluster and fusion methods temporarily
    cluster_methods = ['hierarchical']
    fusion_methods = ['disagreement']

    # Set up our parameters for the genetic algorithm                   Index
    toolbox.register("c_1_type", random.choice, cluster_methods)        # 0
    toolbox.register("c_1_method", random.choice, linkages)             # 1
    toolbox.register("c_1_k", random.randint, k_min, k_max)             # 2
    toolbox.register("c_2_type", random.choice, cluster_methods)        # 3
    toolbox.register("c_2_method", random.choice, linkages)             # 4
    toolbox.register("c_2_k", random.randint, k_min, k_max)             # 5
    toolbox.register("c_3_type", random.choice, cluster_methods)        # 6
    toolbox.register("c_3_method", random.choice, linkages)             # 7
    toolbox.register("c_3_k", random.randint, k_min, k_max)             # 8
    toolbox.register("c_1_pre_type", random.choice, cluster_methods)    # 9
    toolbox.register("c_1_pre_method", random.choice, linkages)         # 10
    toolbox.register("c_1_pre_k", random.randint, k_min, k_max)         # 11
    toolbox.register("c_2_pre_type", random.choice, cluster_methods)    # 12
    toolbox.register("c_2_pre_method", random.choice, linkages)         # 13
    toolbox.register("c_2_pre_k", random.randint, k_min, k_max)         # 14
    toolbox.register("c_3_pre_type", random.choice, cluster_methods)    # 15
    toolbox.register("c_3_pre_method", random.choice, linkages)         # 16
    toolbox.register("c_3_pre_k", random.randint, k_min, k_max)         # 17
    toolbox.register("fusion_method", random.choice, fusion_methods)    # 18

    # How the chromosomes are created
    N_CYCLES = 1
    # TODO: Add user parameter for N_CYCLES
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
    def mutate(individual):
        # Choose a random parameter to mutate, and select the gene
        index = random.randint(0, len(individual) - 1)
        gene = individual[index]

        if gene in [0, 3, 6, 9, 12, 15]:
            individual[gene] = random.choice(cluster_methods)
        elif gene in [1, 4, 7, 10, 13, 16]:
            individual[gene] = random.choice(linkages)
        elif gene in [2, 5, 8, 11, 14, 17]:
            individual[gene] = random.randint(k_min, k_max)
        elif gene == 18:
            individual[gene] = random.choice(fusion_methods)

        return individual,

    def evaluate(individual):

        # sklearn uses np.matrix which throws a warning as it is deprecated
        # Use this to silence the warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sil = parea_2(data,
                            individual[0],
                            individual[1],
                            individual[2],
                            individual[3],
                            individual[4],
                            individual[5],
                            individual[6],
                            individual[7],
                            individual[8],
                            individual[9],
                            individual[10],
                            individual[11],
                            individual[12],
                            individual[13],
                            individual[14],
                            individual[15],
                            individual[16],
                            individual[17],
                            individual[18],
                            fitness=True, k_final=k_final)

        print("Silhouette score: %s" % sil)

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

    # TODO: Work out if pop and log should also be returned
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, stats=stats, halloffame=hall_of_fame, verbose=True)

    print(f"\nSummary:\n{log}")

    return hall_of_fame[0]

def convert_to_parameters(data_len, params):
    """
    A helper function to convert the learned parameters of a genetic algorithm
    search of Parea 2. See the project's Jupyer notebooks on GitHub for details.
    """
    clusterers_index        = [*range(0, data_len)]
    methods_index           = [*range(data_len*1, data_len*2)]
    k_s_index               = [*range(data_len*2, data_len*3)]
    pre_clusterers_index    = [*range(data_len*3, data_len*4)]
    pre_methods_index       = [*range(data_len*4, data_len*5)]
    pre_k_s_index           = [*range(data_len*5, data_len*6)]

    return {"clusterers": list(itemgetter(*clusterers_index)(params)),
            "methods": list(itemgetter(*methods_index)(params)),
            "k_s": list(itemgetter(*k_s_index)(params)),
            "precomputed_clusterers": list(itemgetter(*pre_clusterers_index)(params)),
            "precomputed_methods": list(itemgetter(*pre_methods_index)(params)),
            "precomputed_k_s": list(itemgetter(*pre_k_s_index)(params)),
            "fusion_method": params[-1]}
