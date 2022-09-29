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
c = pyrea.clusterer("ward")

v = pyrea.view(d, c)
```

You now have a view `v`, containing the data `d` using the clustering algorithm
`c`. Note that many views can share the same clusterer, or each view may have a
unique clusterer.

As this is a library for multi-view ensemble learning, you will normally have
multiple views.

A fusion algorithm is therefore used to fuse the clusterings created from
multiple views. Therefore, our next step is to create a *fuser* object:

```python
f = pyrea.fuser('agreement')
```

With you fusion algorithm `f`, you can execute an *ensemble*. The ensemble is
created with a set of views, a fusion algorithm, and a clustering algorithm,
and returns a new view:

```pythom
v_res = pyrea.execute_ensemble([v1, v2, v3], f, c)
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
# algorithm for both views
c = pyrea.clusterer('ward')

# Create the views using the data and the same clusterer
v1 = pyrea.view(d1, c)
v2 = pyrea.view(d1, c)

# Create a fusion object
f = pyrea.fuser('agreement')

# Execute an ensemble based on your views, fusion algorithm, and clusterer
v_res = pyrea.execute_ensemble([v1, v2], f, c)
```

## Ensemble Structures
Complex structures can be built using Pyrea.

For example, examine the two structures below:

![Ensemble Structures](https://raw.githubusercontent.com/mdbloice/AugmentorFiles/master/Pyrea/parea.png)

We will demonstrate how to create deep and flexible ensemble structures using
the examples  a) and b) from the image above.

### Example A
This ensemble consists of two sets of three views, which are clustered, fused,
and then once again combined in a second layer.

We create two ensembles, which represent the first layer of structure a) in
the image above:

```python
import pyrea
import numpy as np

# Clusterers:
hc1 = pyrea.clusterer('ward')
hc2 = pyrea.clusterer('complete')

# Fusion algorithm:
f = pyrea.fuser('agreement')

# Create three random datasets
d1 = np.random.rand(100,10)
d2 = np.random.rand(100,10)
d3 = np.random.rand(100,10)

# Views for ensemble 1
v1 = pyrea.view(d1, hc1)
v2 = pyrea.view(d2, hc1)
v3 = pyrea.view(d3, hc1)

# Execute ensemble 1 and retrieve a new view, which is used later.
v_ensemble_1 = pyrea.execute_ensemble([v1, v2, v3], f, hc1)

# Views for ensemble 2
v4 = pyrea.view(d1, hc2)
v5 = pyrea.view(d2, hc2)
v6 = pyrea.view(d3, hc2)

# Execute our second ensemble, and retreive a new view:
v_ensemble_2 = pyrea.execute_ensemble([v4, v5, v6], f, hc1)

# Now we can execute a further ensemble, using the views generated from the
# two previous ensemble methods:
v_final = pyrea.execute_ensemble([v_ensemble_1, v_ensemble_1], f, hc1)
```

As for structure b) in the image above, this can implemented as follows:

```python
import pyrea
import numpy as np

# Clustering algorithms
c1 = pyrea.clusterer('ward')
c2 = pyrea.clusterer('complete')
c3 = pyrea.clusterer('single')

# Fusion algorithm
f = pyrea.fuser('agreement')

# Create the views with the random data directly:
v1 = pyrea.view(np.random.rand(100,10), c1)
v2 = pyrea.view(np.random.rand(100,10), c2)
v3 = pyrea.view(np.random.rand(100,10), c3)

v_res = pyrea.execute_ensemble([v1, v2, v3], f, [c1, c2, c3])
```

Notice how the ensemble is passed 3 clustering algorithms `[c1, c2, c3]`, and these are combined for a final clustering.

## Extensible

Pyrea has been designed to be extensible. It allows you to use Pyrea's data fusion techniques with custom clustering algorithms that can be loaded in to Pyrea at run-time.

By providing a `View` with a `ClusterMethod` object, it makes providing custom clustering algorithms uncomplicated. See [`Extending Pyrea`](https://pyrea.readthedocs.io/pyrea/extending.html#custom-clustering-algorithms) for details.

# Work In Progress and Future Work
Several features are currently work in progress, future updates will include
the features described in the sections below.

## Genetic Algorithm
Pyrea can select the best clustering algorithms and fusion algorithms based on
a genetic algorithm optimisation technique.

## HCfused Clustering Algorithm
A novel fusion technique, developed by one of the authors of this software
package, named HCfused, will be included soon in a future update.

Details of the HCfused method can be found here:

Pfeifer, Bastian, and Schimek, Michael G. "A hierarchical clustering and data
fusion approach for disease subtype discovery" *Journal of Biomedical
Informatics* **113** (2021): 103636.

<https://www.sciencedirect.com/science/article/pii/S1532046420302641>
