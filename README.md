# Pyrea

<p align="center">
     <img src="https://raw.githubusercontent.com/mdbloice/AugmentorFiles/master/Pyrea/Pyrea-logos_transparent.png" width="400">
</p>
<p align="center">
     Multi-view clustering with custom, executable ensemble graphs.
</p>
<p align="center">
     <a href="https://pypi.org/project/pyrea/"><img src="https://img.shields.io/pypi/v/Pyrea"></a>
     <a href="https://github.com/mdbloice/Pyrea/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/Pyrea"></a>
     <a href="https://github.com/mdbloice/Pyrea/actions/workflows/main.yml"><img src="https://img.shields.io/pypi/pyversions/Pyrea"></a>
</p>
<p align="center">
     <em>The name Pyrea is derived from the Greek word Parea, meaning a group of friends who gather to share experiences, values, and ideas.</em>
</p>

---

## Installation

Install the latest published release from PyPI:

```bash
pip install pyrea
```

The graph API described below is part of the development checkout. To use it
before it is included in a published release, install from source:

```bash
git clone https://github.com/mdbloice/Pyrea.git
cd Pyrea
python -m pip install -e .
```

This installs the Python dependencies, including NetworkX. No C/C++ compilation
or PyTorch installation is needed for the graph workflow.

## Building ensemble graphs

The development version includes a NetworkX-based API for custom ensemble
topologies. Use executable blocks from `pyrea.nn` directly, or compose them in
a class with a `forward()` method:

```python
import numpy as np
from pyrea import nn


class MyEnsemble(nn.Module):
    def __init__(self, n_clusters=3):
        self.rna_cluster = nn.Hierarchical(n_clusters, method='ward')
        self.meth_cluster = nn.Hierarchical(n_clusters, method='complete')
        self.fusion = nn.Fusion('disagreement')
        self.final = nn.Hierarchical(n_clusters, input_kind=nn.DISTANCE)

    def forward(self, rna, methylation):
        first = self.rna_cluster(rna)
        second = self.meth_cluster(methylation)
        return self.final(self.fusion(first, second))


model = MyEnsemble().compile()  # Builds the graph without executing clustering.

# Two views of the same 30 samples, with different numbers of features.
rng = np.random.RandomState(42)
data_rna = rng.normal(size=(30, 8))
data_methylation = rng.normal(size=(30, 5))
labels = model(rna=data_rna, methylation=data_methylation)
assert labels.shape == (30,)

print(model.get_params())
trial = model.with_params(final__n_clusters=4, rna_cluster__method='average')
trial_labels = trial(rna=data_rna, methylation=data_methylation)
```

This follows the `Module`/`forward()` composition style familiar from PyTorch.
Calling a module directly on arrays executes it eagerly; `compile()` records
its block connections as a directed acyclic graph (DAG).

Graphs support branches of different depths, shared intermediate results,
multiple fusion stages, and named outputs. Execution follows dependency order
and computes shared nodes once per invocation. `model.graph` exposes a frozen
NetworkX graph for breadth-first traversal, inspection, and visualization.
Custom operations can subclass `nn.Block` or wrap a callable with `nn.Function`.
No PyTorch dependency is required.

### Functional construction and custom topologies

You can also connect blocks directly using symbolic inputs. This example
creates two branches, fuses their cluster assignments, and exposes both final
labels and the fused distance matrix:

```python
rna = nn.Input('rna')
methylation = nn.Input('methylation')

first = nn.Hierarchical(3, method='ward')(rna)
second = nn.Hierarchical(3, method='average')(methylation)
distance = nn.Fusion('disagreement')(first, second)
final = nn.Hierarchical(3, input_kind=nn.DISTANCE)(distance)

custom = nn.Graph(
    inputs={'rna': rna, 'methylation': methylation},
    outputs={'labels': final, 'distance': distance},
)
result = custom(rna=data_rna, methylation=data_methylation)
assert result['distance'].shape == (30, 30)
```

For reusable custom operations, subclass `nn.Block` and implement `forward()`.
To adapt an existing callable, use `nn.Function` with declared input and output
kinds. Compositions can nest other modules and connect any number of branches
and fusion stages. Cycles are rejected.

All input views must contain the same samples in the same order. Use pandas
DataFrames or `nn.SampleData(array, sample_ids)` to validate identifiers across
identified inputs; bare arrays use positional alignment. Connections distinguish
features, labels, distances, and affinities. For example, disagreement fusion
produces distances, while agreement fusion produces affinities; downstream
blocks must accept the corresponding kind. `nn.Consensus()` returns the strict
consensus labels, which may contain more clusters than its input partitions.

### Graph traversal and parameter search

```python
import networkx as nx

reachable = list(nx.bfs_tree(model.graph, source='rna'))
execution_order = list(nx.topological_sort(model.graph))
```

Breadth-first traversal helps inspect a topology. Execution uses topological
ordering so a block waits for all its inputs, including branches of different
depths. Shared nodes execute once per call, and results are not cached across
calls with new data. Execution is currently sequential.

`get_params()` exposes configuration using `block_name__parameter` keys, as in
the first example. `with_params()` creates an independent configured copy for
an optimization trial. Custom blocks can expose their own parameters through
`get_params()` and `set_params()`. These hooks support parameter search over
custom graphs.

### Genetic optimization of graph configurations

`pyrea.optim.genetic_search()` searches block parameters while keeping the
topology fixed. For the `MyEnsemble` graph above:

```python
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from pyrea import optim

# Define the evaluation geometry once, independently of candidate parameters.
evaluation_features = StandardScaler().fit_transform(
    np.column_stack([data_rna, data_methylation])
)
search = optim.genetic_search(
    model,
    inputs={'rna': data_rna, 'methylation': data_methylation},
    search_space={
        'rna_cluster': {'method': ['ward', 'average'], 'n_clusters': optim.Int(2, 5)},
        'meth_cluster': {'method': ['complete', 'average'], 'n_clusters': optim.Int(2, 5)},
        'final': {'n_clusters': optim.Int(2, 5)},
    },
    fitness=lambda labels: silhouette_score(evaluation_features, labels),
    population_size=12, generations=5, random_state=42,
)
best_labels = search.best_graph(rna=data_rna, methylation=data_methylation)
print(search.best_score, search.best_params)
```

Search spaces use **block names**, with `Choice`, inclusive `Int`, and `Float`
domains. Lists of parameter mappings define conditional cases, allowing a block
to switch clustering families with their respective hyperparameters. The result
includes an independent best graph, evaluation history, and generation logbook.
The original graph is unchanged. Scores are maximized unless `maximize=False`.
Set estimator seeds as well as the search seed for reproducibility. See the
[guide](docs/source/graphs.rst) for conditional spaces and failure handling.

### Train once and assign new samples

Compiled graphs support inductive `fit()` / `predict()` execution. Prediction
accepts one new sample or a batch and uses frozen preprocessing, training
references, and cluster identities. It does not rerun clustering or optimization.

```python
model = MyEnsemble().compile()
model.fit({'rna': data_rna, 'methylation': data_methylation}, assignment='nearest')
new_patient = {
    'rna': data_rna[:1] + 0.1,
    'methylation': data_methylation[:1] + 0.1,
}
assigned_labels = model.predict(new_patient)  # shape (1,)
training_labels = model.labels_
```

For hierarchical, spectral, DBSCAN, and OPTICS blocks, this is an explicit
**frozen-label assignment extension**. The default transfers the label of the
nearest training sample. `assignment='average'` uses average distance to each
training cluster; `'medoid'` uses its learned representative. Rules can also be
set per named clustering block. These rules preserve the training partition;
they do not claim native prediction for a transductive algorithm.

Use `nn.Standardize()` or `nn.Transform(StandardScaler())` for retained training
statistics, and `nn.Estimator(KMeans(...))` for an estimator with native
`predict()`. Custom blocks implement `fit_forward()` and `predict_forward()`.
An arbitrary function or a block that refits preprocessing in `forward()`
cannot automatically become inductive.

Fitted fusion computes new-by-training pairwise matrices. Final consensus keeps
the learned label mapping; unseen membership combinations use the nearest
observed tuple, or raise with `consensus='raise'`. `predict()` returns labels,
while `transform()` exposes all declared outputs and optional intermediates.
`run()` and ordinary graph calls retain their transductive behavior.

After genetic search, call `search.best_graph.fit(training_views)` before
prediction. Parameter copies made with `with_params()` are unfitted. See the
[inductive guide](docs/source/inductive.rst) and
[held-out Nutrimouse notebook](notebooks/Nutrimouse_Inductive.ipynb) for the
assignment definitions, sample schemas, and persistence examples.

### Predefined Parea graphs

```python
from pyrea import models

parea1 = models.parea_1(
    ['rna', 'methylation'], methods=('ward', 'complete'), n_clusters=3
)
parea2 = models.parea_2(
    ['rna', 'methylation'], methods=('ward', 'average'), n_clusters=3,
    return_matrix=True,
)
labels = parea1(rna=data_rna, methylation=data_methylation)
result = parea2(rna=data_rna, methylation=data_methylation)
```

These builders use the same public API available for custom topologies and
support any positive number of input views. Parea 1 takes two initial methods;
Parea 2 takes one per view and defaults to Ward for every view when methods are
omitted. `precomputed_methods` selects methods for subsequent clustering.
`k_final` selects a single final clustering with a fixed number of clusters
instead of strict consensus. `return_matrix=True` returns named `labels` and
`distance` outputs. Both builders use Python disagreement fusion; HCfused is
not used.

### Compatibility

The existing eager API (`pyrea.view`, `pyrea.execute_ensemble`,
`pyrea.parea_1`, and `pyrea.parea_2`) remains available. The new
`pyrea.models.parea_1` and `pyrea.models.parea_2` builders return graphs;
the older top-level helpers execute directly on data.

This update also fixes three existing behaviors: hierarchical clustering now
honors the selected linkage method and optimal ordering, DBSCAN returns its
labels, and fusion/consensus correctly handle SciPy's column-shaped labels.
Results previously affected by these bugs can change.

See [the graph API guide](docs/source/graphs.rst) for functional construction,
custom blocks, sample alignment, parameter search, and compatibility details.

## Demonstration Notebooks

- [Nutrimouse with the new API](notebooks/Nutrimouse_New_API.ipynb): follows the
  original notebook's sequence with hierarchical and spectral Parea 1, three
  genetic searches, comparisons against the same three optional `mvlearn`
  methods, and hierarchical Parea 2. It retains the raw views and original
  search ranges and population/generation settings, fixes every final layer
  at `k_final=2`, and computes the results and selected configurations from
  the current run. Installation notes cover the optional comparison dependency.
- [Inductive Nutrimouse tutorial](notebooks/Nutrimouse_Inductive.ipynb): fits a
  multi-view graph on a training subset, assigns held-out mice individually and
  in batches, verifies unchanged training state, and demonstrates retained
  preprocessing, native prediction, and serialization.
- [Nutrimouse graph tutorial](notebooks/Nutrimouse_Graphs.ipynb): replays the
  original raw-data Parea 2 configuration and runs a fresh genetic search with
  its original parameter ranges, population/generation settings, and
  fused-distance silhouette objective. Separate comparisons vary standardization
  and objective while retaining the Parea 2 topology and search settings. Tables
  report both fused-distance and fixed-view silhouette, plus genotype and diet
  agreement. Further examples cover custom graphs, `Module.forward()`, NetworkX
  traversal, spectral affinities, conditional parameter searches, and fitting
  once to assign a single new sample. The small
  dataset is bundled with provenance, so execution needs no data download or
  `mvlearn`. Scores are computed from the current run; the original saved
  partition verifies the replay independently of cluster numbering.
- [Original Nutrimouse notebook](notebooks/Nutrimouse.ipynb): the earlier eager
  API demonstration using hierarchical and spectral clustering with genetic
  helpers and comparisons against `mvlearn` algorithms. It requires the optional
  `mvlearn` dependency and has not been migrated to the graph API.

To run the new graph tutorial, install the development checkout and the optional
notebook dependencies from the repository root:

```bash
python -m pip install -e .
python -m pip install -r notebooks/requirements.txt
jupyter lab notebooks/Nutrimouse_Graphs.ipynb
```

## Paper and eager API
The Pyrea software package is the accompanying software for our paper:

Pfeifer, B., Bloice, M.D., & Schimek, M.G. (2023). *Parea: multi-view ensemble clustering for cancer subtype discovery*. **Journal of Biomedical Informatics**. <https://doi.org/10.1016/j.jbi.2023.104406>

While Pyrea allows for flexible and custom architectures to be built, two structures are discussed specifically in the paper cited above, namely Parea 1 and Parea 2.

Both the structures, which are described in detail below as well as in the paper mentioned above, can be quickly generated and applied to your data using two helper functions, `parea_1()` and `parea_2()`, and can be quickly run as follows:

```python
import pyrea
import numpy as np

# Create sample data:
d1 = np.random.rand(100,10)
d2 = np.random.rand(100,10)
d3 = np.random.rand(100,10)

data = [d1,d2, d3]

labels = pyrea.parea_2(data)
```

which executes Parea 2.

These helpers use disagreement fusion by default. They implement the ensemble
structures with Python fusion methods; they do not reproduce the paper's
HCfused fusion procedure. Parameters can be overridden, and the existing
genetic helpers search configurations of the predefined eager structures.

### Genetic Algorithm

The existing hierarchical genetic helpers search linkage methods and cluster
counts within the predefined Parea structures, using disagreement fusion.

For example, to find optimal parameters for Parea 2:

```python
import pyrea
from sklearn import datasets

d1 = datasets.load_iris().data
d2 = datasets.load_iris().data
d3 = datasets.load_iris().data

data = [d1,d2,d3]

params = pyrea.parea_2_genetic(data, k_min=2, k_max=5)
```

where `k_min` and `k_max` refer to the minimum and maximum number of clusters to attempt for each layer, respectively.

Note that `params` contains the optimal parameters found by the genetic algorithm. To get the labels, run `parea_2()` passing your data and these optimal parameters:

```python
pyrea.parea_2(data, *params)
```

which will return the cluster labels for your data.

Also, you may choose to define the **final** number of clusters returned by the algorithm (but allowing it to optimise intermediate numbers of clusters) by defining `k_final`, e.g:

```python
params = pyrea.parea_2_genetic(data, k_min=2, k_max=5, k_final=3)
```

and calling `parea_2()` as follows:

```python
pyrea.parea_2(data, *params, k_final=3)
```

The fixed `parea_2_genetic` helper expects three views. The older
`parea_2_mv_genetic` helper currently has a string-formatting failure and is
not used by the graph API. For custom graph configurations, use
`pyrea.optim.genetic_search()` described above. The older helpers retain their
predefined structures.

### API

**Please note that Pyrea is work in progress. The API may change from version
to version and introduce breaking changes.**

In Pyrea, your data are organised in to views. A view consists of the data in
the form of a 2D matrix, and an associated clustering algorithm (a *clusterer*).

To create a view you must have some data, and a clusterer:

```python
import pyrea
import numpy as np

# Create your data, which must be a 2-dimensional array/matrix.
d = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])

# Create a clusterer
c = pyrea.clusterer("hierarchical", n_clusters=2, method='ward')

v = pyrea.view(d, c)
```

You now have a view `v`, containing the data `d` using the clustering algorithm
`c`. Note that many views can share the same clusterer, or each view may have a
unique clusterer.

To obtain the cluster solution the specified view can be executed

```python
v.execute()
```

The clustering algorithm can be either 'spectral', 'hierarchical', 'dbscan', or 'optics'. See the documentation for a complete list of parameters that can be passed when creating a clusterer.

As this is a library for multi-view ensemble learning, you will normally have
multiple views.

A fusion algorithm is therefore used to fuse the clusterings created from
multiple views. Therefore, our next step is to create a *fuser* object:

```python
f = pyrea.fuser('disagreement')
```

With you fusion algorithm `f`, you can execute an *ensemble*. The ensemble is created with a set of views and a fusion algorithm,
and its returned object (distance or affinity matrix) can again be specified as a view:

```python
# Create a new clusterer with precomputed=True
c_pre = pyrea.clusterer("hierarchical", n_clusters=2, method='ward', precomputed=True)
v_res = pyrea.view(pyrea.execute_ensemble([v1, v2, v3], f), c_pre)
```

This newly created view, `v_res` can subsequently be fed into another ensemble,
allowing you to create stacked ensemble architectures, with high flexibility.

A full example is shown below, using random data:

```python
import pyrea
import numpy as np

# Create two datasets with random values of 1000 samples and 100 features per sample.
d1 = np.random.rand(1000,100)
d2 = np.random.rand(1000,100)

# Define the clustering algorithm(s) you want to use. In this case we used the same
# algorithm for both views. By default n_clusters=2.
c = pyrea.clusterer('hierarchical', n_clusters=2, method='ward')

# Create the views using the data and the same clusterer
v1 = pyrea.view(d1, c)
v2 = pyrea.view(d2, c)

# Create a fusion object
f = pyrea.fuser('disagreement')

# Specify a clustering algorithm (precomputed = True)
c_pre = pyrea.clusterer("hierarchical", n_clusters=2, method='ward', precomputed=True)
# Execute an ensemble based on your views and a fusion algorithm
v_res = pyrea.view(pyrea.execute_ensemble([v1, v2], f), c_pre)

# The cluster solution can be obtained as follows
v_res.execute()
```

## Ensemble Structures (eager API)
Complex structures can be built using Pyrea.

For example, examine the two structures below:

![Ensemble Structures](https://raw.githubusercontent.com/mdbloice/AugmentorFiles/master/Pyrea/parea.png)

We will demonstrate how to create deep and flexible ensemble structures using
the examples a) and b) from the image above.

### Example A
This ensemble consists of two sets of three views, which are clustered, fused,
and then once again combined in a second layer.

We create two ensembles, which represent the first layer of structure a) in
the image above:

```python
import pyrea
import numpy as np

# Clusterers:
hc1 = pyrea.clusterer('hierarchical', method='ward', n_clusters=2)
hc2 = pyrea.clusterer('hierarchical', method='complete', n_clusters=2)

# Fusion algorithm:
f = pyrea.fuser('disagreement')

# Create three random datasets
d1 = np.random.rand(100,10)
d2 = np.random.rand(100,10)
d3 = np.random.rand(100,10)

# Views for ensemble 1
v1 = pyrea.view(d1, hc1)
v2 = pyrea.view(d2, hc1)
v3 = pyrea.view(d3, hc1)

# Execute ensemble 1 and retrieve a new view, which is used later.
hc1_pre = pyrea.clusterer('hierarchical', method='ward', n_clusters=2, precomputed=True)
v_ensemble_1 = pyrea.view(pyrea.execute_ensemble([v1, v2, v3], f), hc1_pre)

# Views for ensemble 2
v4 = pyrea.view(d1, hc2)
v5 = pyrea.view(d2, hc2)
v6 = pyrea.view(d3, hc2)

# Execute our second ensemble, and retreive a new view:
hc2_pre = pyrea.clusterer('hierarchical', method='complete', n_clusters=2, precomputed=True)
v_ensemble_2 = pyrea.view(pyrea.execute_ensemble([v4, v5, v6], f), hc2_pre)

# Now we can execute a further ensemble, using the views generated from the
# two previous ensemble methods:
d_fuse  = pyrea.execute_ensemble([v_ensemble_1, v_ensemble_2], f)

# The returned distance matrix is now used as an input for the two clustering methods (hc1 and hc2)
v1_fuse = pyrea.view(d_fuse, hc1_pre)
v2_fuse = pyrea.view(d_fuse, hc2_pre)

# and the cluster solutions are combined
pyrea.consensus([v1_fuse.execute(), v2_fuse.execute()])
```

#### Helper Function
See the `parea_1()` helper function for a pre-built version of structure above.

### Example B
As for structure b) in the image above, this can implemented as follows:

```python
import pyrea
import numpy as np

# Clustering algorithms
c1 = pyrea.clusterer('hierarchical', method='ward', n_clusters=2)
c2 = pyrea.clusterer('hierarchical', method='complete', n_clusters=2)
c3 = pyrea.clusterer('hierarchical', method='single', n_clusters=2)

# Clustering algorithms (so it works with a precomputed distance matrix)
c1_pre = pyrea.clusterer('hierarchical', method='ward', n_clusters=2, precomputed=True)
c2_pre = pyrea.clusterer('hierarchical', method='complete', n_clusters=2, precomputed=True)
c3_pre = pyrea.clusterer('hierarchical', method='single', n_clusters=2, precomputed=True)

# Fusion algorithm
f = pyrea.fuser('disagreement')

# Create the views with the random data directly:
v1 = pyrea.view(np.random.rand(100,10), c1)
v2 = pyrea.view(np.random.rand(100,10), c2)
v3 = pyrea.view(np.random.rand(100,10), c3)

# Create the ensemble and define new views based on the returned disagreement matrix v_res
v_res  = pyrea.execute_ensemble([v1, v2, v3], f)
v1_res = pyrea.view(v_res, c1_pre)
v2_res = pyrea.view(v_res, c2_pre)
v3_res = pyrea.view(v_res, c3_pre)

# Get the final cluster solution
pyrea.consensus([v1_res.execute(), v2_res.execute(), v3_res.execute()])
```

#### Helper Function
See the `parea_2()` helper function for a pre-built version of structure above.

## Extensible

For graph-based ensembles, subclass `nn.Module` to define reusable topologies,
subclass `nn.Block` for custom executable operations, or use `nn.Function`
to adapt a callable. See the [graph API guide](docs/source/graphs.rst).

The eager API also supports custom `Clusterer` implementations associated with
a `View`. See [`Extending Pyrea`](https://pyrea.readthedocs.io/en/latest/extending.html)
for that interface.

## Future work

### Graph-topology search

Genetic optimization supports the parameters of custom graphs with fixed
connectivity. Evolving graph connectivity remains future work.
The historical HCfused C++ source remains in the repository for
reference; integrating it is not part of the current direction.

## Tests

After installing the development checkout, run:

```bash
python -m pip install pytest
python -m pytest tests -q
```

The suite covers custom DAG execution, shared branches, repeated calls,
connection and sample alignment checks, parameter trials, clustering methods,
and Parea graph compatibility. GitHub Actions is configured to install the
checkout and run these tests on Python 3.8–3.11 with Ubuntu 22.04. The graph
update has also been validated locally on Python 3.11 with both the prepared
dependency stack and current dependency versions.

## Miscellaneous

Logo made by Adobe Express Logo Maker: <https://www.adobe.com/express/create/logo>
