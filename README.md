# Pyrea

<p align="center">
<img src="https://raw.githubusercontent.com/mdbloice/AugmentorFiles/master/Pyrea/Pyrea-logos_transparent.png" width="400">
</p>

Multi-view clustering with flexible ensemble structures.

_The name Pyrea is derived from the Greek word Parea, meaning a group of friends who gather to share experiences, values, and ideas._

![PyPI](https://img.shields.io/pypi/v/Pyrea) ![PyPI - License](https://img.shields.io/pypi/l/Pyrea) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Pyrea)

## Installation

Install Pyrea using `pip`:

```bash
pip install pyrea
```

This will install the latest version of Pyrea from PyPI.

## Usage

### API

In Pyrea, your data are organised in to views. A view consists of the data in the
form of a 2D matrix, and an associated clustering algorithm (a _clusterer_).

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

A fusion algorithm is therefore used to fused the clusters created from multiple
views. Therefore, our next step is to create a *fuser* object:

```python
f = pyrea.fuser('parea')
```

With you fusion algorithm `f`, you can create an *ensemble*. The ensemble is
created with your views, the fusion algorithm, and a clustering algorithm:

```pythom
e = pyrea.ensemble([v1, v2, v3], f, c)
```

To perform this operation you execute the ensemble:

```python
e.execute()
```

which returns your clustered fusion matrix.

A full example is shown below:

```python
import pyrea
import numpy as np

# Create two datasets with random values of 1000 samples and 100 features per sample.
d1 = np.random.rand(1000,100)
d2 = np.random.rand(1000,100)

# Define the clustering algorithm(s) you want to use. In this case we used the same
# algorithm for both views
c = pyrea.clusterer('ward')

# Create the views using the data and clusterer
v1 = pyrea.view(d1, c)
v2 = pyrea.view(d1, c)

# Create a fusion object
f = pyrea.fuser('parea')

# Create your ensemble
e = pyrea.ensemble([v1, v2], f, c)

# Execute the ensemble
e.execute()
```

## Ensemble Structures
Complex structures can be built using Pyrea.

For example, examine the two structures below:

![Ensemble Structures](https://raw.githubusercontent.com/mdbloice/AugmentorFiles/master/Pyrea/parea.png)

We will demonstrate how to create deep and flexible ensemble structures using the examples  a) and b) from the image above.

### Example A
This ensemble consists of two sets of three views, which are clustered, fused, and then once again combined in a second layer.

**Work in progress**

We create two ensembles, which represent the first layer of the structure A in the image above:

```python
import pyrea
import numpy as np

# Define our clustering algorithms(s) and fusion algorimth(s) first that we
# will use multiple times.
# Clusterers:
hc1 = pyrea.clusterer('ward')
hc2 = pyrea.clusterer('complete')
# Fusers:
parea_fuser = pyrea.fuser('parea')

# Data (random data for now)
d1 = np.random.rand(100,10)
d2 = np.random.rand(100,10)
d3 = np.random.rand(100,10)

# Views for ensemble 1
view1_e1 = pyrea.view(d1, hc1)
view2_e1 = pyrea.view(d2, hc1)
view3_e1 = pyrea.view(d3, hc1)

# Ensemble 1
e1 = pyrea.ensemble([view1_e1, view2_e1, view3_e1], f, hc1)

# Views for ensemble 2
view1_e2 = pyrea.view(d1, hc2)
view2_e2 = pyrea.view(d2, hc2)
view3_e2 = pyrea.view(d3, hc2)

e2 = pyrea.ensemble([view1_e2, view2_e2, view3_e2], f, hc1)

e3 = pyrea.ensemble([e1, e2], f, [c1, c2])

e3.execute()
```

As for structure b) above, this can implemented as follows:

```python
import pyrea
import numpy as np

# Clustering algorithms
c1 = pyrea.clusterer('ward')
c2 = pyrea.clusterer('complete')
c3 = pyrea.clusterer('single')

# Fusion algorithm
fuser = pyrea.fuser('disagreement')

# Data
d1 = np.random.rand(100,10)
d2 = np.random.rand(100,10)
d3 = np.random.rand(100,10)

v1 = pyrea.view(d1, c1)
v2 = pyrea.view(d2, c2)
v3 = pyrea.view(d3, c3)

e = pyrea.ensemble([v1, v2, v3], f, [c1, c2, c3])

e.execute()
```

Notice how the ensemble is passed 3 clustering algorithms, and these are combined for a final clustering.

## Deep Ensembles
Pyrea can be used to create deep ensembles.

## Genetic Algorithm
Pyrea can select the best clustering algorithms and fusion algorithms based on a genetic algorithm optimisation technique.

**Work in progress...**

## Clustering Methods

See [`scipy`](https://docs.scipy.org/doc/scipy/reference/cluster.html) and [`sklearn`](https://scikit-learn.org/stable/modules/clustering.html) for details.

## Extensible

Pyrea has been designed to be extensible. It allows you to use Pyrea's data fusion techniques with custom clustering algorithms that can be loaded in to Pyrea at run-time.

By providing a `View` with a `ClusterMethod` object, it makes providing custom clustering algorithms uncomplicated. See [`Extending Pyrea`](https://pyrea.readthedocs.org/pyrea/extending.html#custom-clustering-algorithms) for details.
