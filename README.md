# Pyrea

<p align="center">
<img src="https://raw.githubusercontent.com/mdbloice/AugmentorFiles/master/Pyrea/Pyrea-logos_transparent.png" width="400">
</p>

Multi-view clustering with flexible ensemble structures.

*The name Pyrea is derived from the Greek word Parea, meaning a group of
friends who gather to share experiences, values, and ideas.*

![PyPI](https://img.shields.io/pypi/v/Pyrea) ![PyPI - License](https://img.shields.io/pypi/l/Pyrea) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Pyrea)

## Installation

Install Pyrea using `pip`:

```bash
pip install pyrea
```

This will install the latest version of Pyrea from PyPI.

## Usage

### API

**Please note that Pyrea is work in progress. The API may change from version
to version in the coming weeks, which may introduce breaking changes to legacy
code.**

In Pyrea, your data are organised in to views. A view consists of the data in
the form of a 2D matrix, and an associated clustering algorithm (a *clusterer*).

To create a view you must have some data, and a clusterer:

```python
import pyrea

# Create your data, which must be a 2-dimensional array/matrix.
d = [[1,2,3],
     [4,5,6],
     [7,8,9]]

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

## Ensemble Structures
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

## Genetic Algorithm
The Parea 1 and Parea 2 structures can be optimised using a genetic algorithm in order to find the best combinations of clustering methods, fusion methods, and number of clusters.

For example:

```python
import pyrea
import numpy as np

# Create your data
d1 = np.random.rand(100,10)
d2 = np.random.rand(100,10)
d3 = np.random.rand(100,10)

data = [d1, d2, d3]

pyrea.parea_1_genetic(data, max_k=4)
```


## Extensible
Pyrea has been designed to be extensible. It allows you to use Pyrea's data fusion techniques with custom clustering algorithms that can be loaded in to Pyrea at run-time.

By providing a `View` with a `ClusterMethod` object, it makes providing custom clustering algorithms uncomplicated. See [`Extending Pyrea`](https://pyrea.readthedocs.io/pyrea/extending.html#custom-clustering-algorithms) for details.

# Work In Progress and Future Work
Several features are currently work in progress, future updates will include
the features described in the sections below.

## HCfused Clustering Algorithm
A novel fusion technique, developed by one of the authors of this software
package, named HCfused, will be included soon in a future update.

## General Genetic Optimisation
The package will be extended to allow for any custom Pyrea structures to be optimised using a genetic algorithm.

# Miscellaneous

Logo made by Adobe Express Logo Maker: <https://www.adobe.com/express/create/logo>

# Citing

If you find this package useful and wish to cite it, you can use the following:

Pfeifer, B., Bloice, M.D., & Schimek, M.G. (2022). Parea: multi-view ensemble clustering for cancer subtype discovery. arXiv. <https://arxiv.org/abs/2209.15399>
