# Pyrea
Multi-view clustering with flexible ensemble structures.

_The name Pyrea is derived from the Greek word Parea, meaning a group of friends who gather to share experiences, values, and ideas._

## Installation

Install Pyrea using `pip`:

```bash
pip install pyrea
```

This will install the latest version of Pyrea from PyPI.

## Usage

### API

In Pyrea, you data are organised in to views. A view consists of the data in the
form of a 2D matrix, and an associated clusterer.

To create a view you must have some data, and a clusterer:

```python
import pyrea

# Create a random 3x3 matrix as your data
d = np.random.rand(3,3)

d = [[1,2,3],
     [4,5,6],
     [7,8,9]]

# Create a clusterer
c = pyrea.clusterer("ward", **params)

v = pyrea.View(d, c)
```
You now have a view `v`, containing the data `d` using the clustering algiorithm
`c`.

Many views can share the same clusterer. Or each view may have a unique clusterer.

```python
# Define clustering techniques
c1 = pyrea.cluster.KMeans()
c2 = pyrea.cluster.XYZ()
c3 = pyrea.cluster.XYZ()

# Make some datasets
d1 = np.random.rand(3,3)
d2 = np.random.rand(3,3)
d3 = np.random.rand(3,3)

# A view consists of a dataset (2d array/matrix) and a clustering algorithm
v1 = pyrea.View(d1, c1)
v2 = pyrea.View(d2, c2)
v3 = pyrea.View(d3, c3)

# A fuser is instantiated with a clustering algorithm also
f = pyrea.fusion.consensus(c1)

# An ensemble consists of views and a fusion algorithm:
e1 = Ensemble([v1,v2,v3], f)
e2 = Ensemble([v1,v2,v3], f)

# An ensemble can also be made from ensembles instead of views
e3 = Ensemble([e1, e2], f)

e3.execute()  # returns a final cluster
```

### Second Attempt at API

Import Pyrea as follows:

```python
import pyrea
```

Once Pyrea has been imported, you begin by creating views. Each view has an associated clustering method, which must be defined when creating the view:

```python
print(pyrea.CLUSTER_METHODS)
# ['ward', 'complete', 'single', 'average']

c = pyrea.ClusterMethod("ward")
v = View(data, clustermethod=c)

v_r = v.execute()
```

To view the clustering algorithms you can use, print

By providing a `View` with a `ClusterMethod` object, it makes providing custom clustering algorithms uncomplicated. See [`Extending Pyrea`](https://pyrea.readthedocs.org/pyrea/extending.html#custom-clustering-algorithms) for details.


### First Attempt at API

Import Pyrea as follows:

```python
import pyrea
```

Once Pyrea has been imported, we can begin to load our _views_. A view can be a NumPy matrix, a Python list of lists, or a Pandas DataFrame.

To load a view:

```python
import pandas as pd
from numpy import genfromtxt

view1 = np.genfromtxt('view1.csv', delimiter=';')
view2 = pd.read_csv('view2.csv')

pyrea_view1 = pyrea.load_view(view1)  # Returns a View object
pyrea_view2 = pyrea.load_view(view2)
```

Now that we have our views (which are of type `View`), we can cluster them and then fuse them for later analysis:

```python
fused = pyrea.fuse(pyrea_view1, pyrea_view2)
```

There are several parameters that can be set, including the type of clustering methods you wish to use.

## Workflows
Pyrea is built around the concept of *workflows*, a flexible way to create ensembles with different views, clustering techniques, and fusion techniques.

An example workflow consisting of two views can be seen below:

```python
view1 = pyrea.View(pandas.read_csv('view1.csv'))
view2 = pyrea.View(pandas.read_csv('view2.csv'))

method = pyrea.Method(pyrea.parea_hc(k=3))
cluster = pyrea.Cluster([view1, view2], method)
fuse = pyrea.Fuse(parea_hc(args))

w = pyrea.Workflow(view1, view2, cluster, fuse)
w.execute()
```

Of course, workflows can be much more complex than this.

## Alternative Usage

Alternatively, we create a `ClusterFuser` object, and compartmentalise the data in to this object. For example:

```python

view3 = [[2, 4, 1, ..., 9],
         [4, 8, 8, ..., 0],
         ...,
         [1, 9, 6, ..., 7]]

f = pyrea.ClusterFuser()
f.load_view(view1)
f.load_view([view2, view3])  # Also accepts a list of views.

f.fuse()
```

Note that `ClusterFuser` can accept `View` parameters during initialisation.

## Clustering Methods

See [`scipy`](https://docs.scipy.org/doc/scipy/reference/cluster.html) and [`sklearn`](https://scikit-learn.org/stable/modules/clustering.html) for details.

## Extensible

Pyrea has been designed to be extensible. It allows you to use Pyrea's data fusion techniques with custom clustering algorithms that can be loaded in to Pyrea at run-time.
