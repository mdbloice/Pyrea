# Pyrea
Multi-view hierarchical clustering with flexible ensemble structures.

_The name Pyrea is derived from the Greek word Parea, meaning a group of friends who gather to share experiences, values, and ideas._

## Installation

Install Pyrea using `pip`:

```bash
pip install pyrea
```

This will install the latest version of Pyrea from PyPI.

## Usage

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