Graph-based ensembles
=====================

``pyrea.nn`` provides executable blocks that can be connected into custom
ensemble topologies. A block call on arrays executes immediately. A block call
on symbolic inputs records a dependency instead. ``Graph`` compiles those
dependencies into a NetworkX directed acyclic graph and executes them when
sample data are supplied.

The API follows the ``Module`` / ``forward`` composition style familiar from
PyTorch. It does not require PyTorch. Clustering operations do not use
backpropagation, and compiling a graph does not optimize its parameters.

Building your own topology
--------------------------

This example clusters two views, fuses their labels into distances, and clusters
the resulting distance matrix. No clustering runs while constructing it.

.. code-block:: python

   import numpy as np
   from pyrea import nn

   rna = nn.Input('rna')
   methylation = nn.Input('methylation')

   a = nn.Hierarchical(n_clusters=3, method='ward', name='rna_cluster')(rna)
   b = nn.Hierarchical(n_clusters=3, method='complete', name='meth_cluster')(methylation)
   distance = nn.Fusion('disagreement', name='fusion')(a, b)
   labels = nn.Hierarchical(
       n_clusters=3, input_kind=nn.DISTANCE, name='final'
   )(distance)

   model = nn.Graph(
       inputs={'rna': rna, 'methylation': methylation},
       outputs={'labels': labels, 'distance': distance},
   )

   rng = np.random.RandomState(42)
   data_rna = rng.normal(size=(30, 8))
   data_methylation = rng.normal(size=(30, 5))
   result = model(rna=data_rna, methylation=data_methylation)
   assert result['labels'].shape == (30,)
   assert result['distance'].shape == (30, 30)

You can connect a result to several later blocks, merge branches of different
depths, stack multiple fusion stages, or select several outputs. The graph has
no fixed number of views, branches, or layers. Every declared input must
contribute to an output. Output values can be a single symbolic node or a
nonempty dictionary of named nodes. Each block call produces one value; use
separate blocks when an operation needs multiple output kinds.

Reusable modules
----------------

Subclass ``Module`` to build a reusable composition. Define its blocks in
``__init__`` and connect them in ``forward``. Ordinary Python modules, lists,
and dictionaries of blocks can be nested to organize larger topologies.

.. code-block:: python

   class TwoViewEnsemble(nn.Module):
       def __init__(self, n_clusters=3):
           self.rna_cluster = nn.Cluster('hierarchical', n_clusters=n_clusters, method='ward')
           self.meth_cluster = nn.Hierarchical(n_clusters, method='complete')
           self.fusion = nn.Fusion('disagreement')
           self.final = nn.Hierarchical(n_clusters, input_kind=nn.DISTANCE)

       def forward(self, rna, methylation):
           first = self.rna_cluster(rna)
           second = self.meth_cluster(methylation)
           return self.final(self.fusion(first, second))

   module = TwoViewEnsemble()
   model = module.compile()
   labels = model(data_rna, data_methylation)
   assert np.array_equal(labels, module(data_rna, data_methylation))

``compile()`` infers feature inputs from required, named ``forward`` arguments.
For other kinds or variable signatures, supply explicit inputs, for example
``module.compile(rna=nn.Input('rna'), methylation=nn.Input('methylation'))``.
Tracing follows the Python path taken by symbolic inputs; branches based on
sample values and loops whose length depends on sample values are not traced.
Loops over a configured number of layers or views are supported.

Custom executable blocks
------------------------

Subclass ``Block`` for an indivisible operation. Declare the input kinds by
``forward`` argument name and its output kind. Configuration belongs in the
constructor. ``Module`` composes blocks; ``Block`` records an executable node.

.. code-block:: python

   class Standardize(nn.Block):
       input_kinds = {'data': nn.FEATURES}
       output_kind = nn.FEATURES

       def forward(self, data):
           scale = data.std(axis=0)
           scale[scale == 0] = 1
           return (data - data.mean(axis=0)) / scale

   source = nn.Input('data')
   standardized = Standardize()(source)
   output = nn.Hierarchical()(standardized)
   custom = nn.Graph({'data': source}, output)
   labels = custom(data=data_rna)

For existing functions or external estimators, ``Function`` adapts a callable:

.. code-block:: python

   from sklearn.cluster import KMeans

   def kmeans(data):
       # Fit a fresh estimator per call rather than retaining fitted state.
       return KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(data)

   block = nn.Function(kmeans, {'data': nn.FEATURES}, output_kind=nn.LABELS)
   custom = nn.Graph({'data': source}, block(data=standardized))
   labels = custom(data=data_rna)

Connections bind to the actual callable argument names, including keyword-only
arguments. A variadic argument declaration applies to every supplied input.
Passing the same source to multiple ports is supported. Input kinds omitted
from a custom block's declarations accept any supported kind; declare them
explicitly wherever possible. Custom blocks must preserve sample count and
sample order, treat their incoming arrays as read-only, and avoid retaining
per-run results. The executor does not copy arrays between branches or reset
state deliberately retained by a custom callable.

Kinds and sample alignment
--------------------------

The four kinds are ``FEATURES`` (sample-by-feature matrix), ``LABELS`` (one
label per sample), ``DISTANCE`` and ``AFFINITY`` (sample-by-sample matrices).
Connections with incompatible declared kinds fail during graph construction.
Execution validates sample counts and output shapes. Pairwise matrices must
be square, finite, symmetric, and nonnegative; distances also need a zero
diagonal. Column-shaped labels returned by SciPy are normalized to vectors.

``Fusion('disagreement')`` returns distances. ``Fusion('agreement')`` returns
affinities. ``Fusion('consensus')`` returns a binary affinity matrix, whereas
``Consensus()`` returns final labels. Final consensus uses unanimous
co-membership, so it may produce more clusters than its input partitions.

Hierarchical clustering, DBSCAN, and OPTICS accept ``input_kind=nn.DISTANCE``;
spectral clustering accepts ``input_kind=nn.AFFINITY``. All support feature
inputs. A conversion between affinity and distance is explicit, for example
``nn.AffinityToDistance()(affinity_node)``. That block computes
``max(A) - A``, scales by ``max(A)`` by default, and zeros the diagonal.

All views must describe the same samples in the same order. Array inputs use
positional alignment, which cannot detect swapped rows. Supply
``nn.SampleData(array, sample_ids)`` or pandas DataFrames to check identifiers
across identified inputs. Identifiers must be unique. Pairwise DataFrame
columns must match their row index. Inputs without identifiers still use
positional alignment. Custom operations remain responsible for preserving
that order internally.

Traversal and execution
-----------------------

.. code-block:: python

   import networkx as nx

   topology = model.graph
   reachable = list(nx.bfs_tree(topology, source='rna'))
   execution_order = list(nx.topological_sort(topology))
   labels, intermediates = model.run(
       {'rna': data_rna, 'methylation': data_methylation},
       return_intermediates=True,
   )

``model.graph`` is a frozen NetworkX copy with string node names, block/kind
node attributes, and input port names on edges. Use it for traversal and
visualization; create or rebuild topologies through symbolic block calls.
Cycles are rejected. Breadth-first traversal is useful for inspecting the
network, but execution uses topological ordering so every dependency is ready
before its consumer runs. Execution is currently sequential. Shared nodes run
once per invocation; separate calls to the same block are separate executable
nodes with shared configuration. Results are local to the run, so changed
data always trigger fresh computation. A failing operation reports its node
name and preserves the original exception as its cause.

Parameter search
----------------

.. code-block:: python

   print(model.get_params())
   trial = model.with_params(final__n_clusters=4, rna_cluster__method='average')
   trial_labels = trial(data_rna, data_methylation)

Parameters use ``block_name__parameter`` keys. Compiled modules use attribute
paths, such as ``branch.cluster__n_clusters``. Explicit block names take
precedence. Reusing a block shares its configuration across its calls.
``with_params`` deep-copies the model for an independent trial and leaves the
original configuration unchanged. Custom blocks can expose configuration by
implementing ``get_params()`` and ``set_params(**params)``. Such blocks must
support deep copying; callable closures should not hide mutable trial state.

``Cluster`` exposes the clustering family as ``clusterer`` and accepts its
parameters as keyword arguments. For example,
``nn.Cluster('spectral', n_clusters=3, random_state=42)``. Switching a generic
cluster block's family starts with that family's default parameters; specify
the desired parameters in the same trial. Changes must retain compatible
connection kinds. Changing a fusion output from distance to affinity requires
rebuilding the graph with suitable downstream blocks.

These hooks support ``pyrea.optim.genetic_search`` and tools such as
scikit-learn's ``ParameterGrid``. Choose an explicit evaluation output and
metric, and set random seeds for stochastic clusterers. When comparing
candidate configurations, consider using a fixed evaluation geometry defined
independently of their parameters. Silhouette requires features or distances;
do not pass affinity weights as distances.

Genetic optimization
--------------------

``genetic_search`` optimizes configurations of any compiled graph, including
custom topologies and shared blocks. Connectivity stays fixed. Search spaces
map **block names** from ``named_blocks()`` to parameter mappings. A list of
mappings defines conditional cases, keeping algorithm-specific parameters
together during crossover and mutation:

.. code-block:: python

   from sklearn.metrics import silhouette_score
   from sklearn.preprocessing import StandardScaler
   from pyrea import optim

   evaluation_features = StandardScaler().fit_transform(
       np.column_stack([data_rna, data_methylation])
   )
   space = {
       'rna_cluster': [
           {'clusterer': 'hierarchical', 'method': ['ward', 'average'],
            'n_clusters': optim.Int(2, 5)},
           {'clusterer': 'spectral', 'affinity': 'rbf',
            'gamma': optim.Float(0.001, 0.1, log=True),
            'n_clusters': optim.Int(2, 5), 'random_state': 42},
       ],
       'meth_cluster': {'method': ['complete', 'average']},
       'final': {'n_clusters': optim.Int(2, 5)},
   }
   original_params = model.get_params()
   search = optim.genetic_search(
       model, {'rna': data_rna, 'methylation': data_methylation}, space,
       fitness=lambda labels: silhouette_score(evaluation_features, labels),
       population_size=12, generations=5, random_state=42,
   )
   best_labels = search.best_graph(rna=data_rna, methylation=data_methylation)
   assert model.get_params() == original_params
   print(search.best_score, search.best_params)

``Choice([...])`` and bare lists/tuples specify categorical choices; other
scalars specify fixed values. ``Int(low, high)`` includes both bounds.
``Float(low, high, log=True)`` samples positive values on a logarithmic scale.
Use ``Choice([[...], ...])`` when individual parameter values are themselves
lists. Each candidate's active parameters override a fresh copy of the
original configuration. A family switch starts with the new family's defaults;
include all desired settings for that family in its case. Other blocks retain
their original parameters. Shared blocks receive one configuration across all
their calls. Custom blocks participate through ``get_params`` / ``set_params``.

The fitness callable receives exactly what ``graph.run(inputs)`` returns: a
single value or a dictionary for named outputs. It must return a finite real
scalar. Set ``maximize=False`` for a loss. Labels for external validation are
not required by the optimizer; choosing an objective remains the caller's
responsibility. The Nutrimouse notebook starts with the original Parea 2
fused-distance objective on the supplied measurements. Separate comparisons
change standardization and a fixed, label-free view-distance objective while
holding topology and search settings constant. Both silhouettes and genotype/diet
agreement are reported; the annotations do not enter either fitness callable.
The historical configuration is replayed separately from fresh genetic searches.

Evolution uses tournament selection, crossover, mutation, and elitism. A
mutation resamples one active parameter or switches a block's conditional case;
crossover exchanges compatible parameters or whole cases. ``population_size``
sets the initial population and ``generations`` counts subsequent generations.
With default ``elitism=1``, evaluation requests total
``population_size + generations * (population_size - 1)``. Each candidate runs
sequentially on its own graph copy. Elites retain their recorded scores.
``random_state`` controls only the evolutionary RNG; seed stochastic estimators
and objectives separately. No global random seeds or DEAP creator classes are
modified.

The result exposes:

* ``best_graph``: a configured copy ready to run, without retained trial state.
* ``best_params``: active overrides using ``block__parameter`` keys, suitable
  for ``original.with_params(**search.best_params)``.
* ``best_score``: the best observed finite fitness.
* ``history``: one dictionary per evaluation request, with generation,
  individual index, parameters, score, status, cache flag, failure phase,
  and error message. Failed scores are ``None``.
* ``logbook``: a DEAP logbook with per-generation evaluations, cache hits,
  valid/failed population counts, generation best, overall best, and mean score.
* ``n_evaluations``: actual attempts, excluding cache hits and retained elites.

By default, candidate configuration, execution, or fitness errors raise
``SearchError``, which preserves the original cause, history, and completed
generation summaries. ``on_error='penalize'`` records candidate failures and
assigns worst fitness so search can continue. If every attempt fails, the
search raises ``SearchError`` rather than returning a failed candidate.

``cache=True`` reuses scores, including failures, for identical active parameter
overrides. It requires picklable parameter values and deterministic operations
and fitness. The default is ``False``, so duplicate candidates are reevaluated.
An unseeded stochastic graph may produce a different score when the returned
best graph is run again. Custom operations and fitness must treat input data
and outputs as read-only; graph copies do not isolate mutable callable closures.
The search changes configurations only; evolving graph edges is not included.

Predefined Parea topologies
---------------------------

.. code-block:: python

   from pyrea import models

   parea1 = models.parea_1(['rna', 'methylation'], methods=('ward', 'complete'))
   parea2 = models.parea_2(
       ['rna', 'methylation'], methods=('ward', 'average'), return_matrix=True
   )
   labels = parea1(data_rna, data_methylation)
   result = parea2(data_rna, data_methylation)

Both builders use the same public block API available to custom topologies,
use disagreement fusion, and support any positive number of input views.
``precomputed_methods`` configures subsequent clustering methods.
``n_clusters`` sets the initial count; individual counts can be changed through
``with_params``. ``k_final`` selects one final clustering with that fixed count
instead of strict consensus. ``return_matrix=True`` exposes ``labels`` and
``distance`` as named outputs. Parea 1 requires two methods; Parea 2 requires
one initial method per view and allows any positive number of final branches.
Its unspecified initial methods default to Ward for every view.

Existing eager functions such as ``pyrea.parea_1`` and ``pyrea.parea_2`` remain
available. The graph implementation also corrects existing linkage selection,
DBSCAN's missing return, and strict consensus handling of column-shaped SciPy
labels. Existing results affected by those bugs can change. HCfused is not
used by the graph API; its existing source files are left in place.

API reference
-------------

.. automodule:: pyrea.graph
   :members: Input, Node, Module, Block, Graph, SampleData

.. automodule:: pyrea.nn
   :members: Cluster, Hierarchical, Fusion, Consensus, Function, AffinityToDistance

.. automodule:: pyrea.models
   :members:

.. automodule:: pyrea.optim
   :members: Choice, Int, Float, SearchResult, SearchError, genetic_search
