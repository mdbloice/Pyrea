.. Parea documentation master file, created by
   sphinx-quickstart on Wed Aug 10 12:50:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Pyrea's documentation!
=================================
Pyrea is a Python package for multi-view hierarchical clustering with flexible ensemble structures.

*The name Pyrea is derived from the Greek word Parea, meaning a group of friends who gather to share experiences, values, and ideas.*

Pyrea is licensed under the terms of the MIT License. See the license section below for details.

Installation is via *pip*:

.. code-block:: python

   pip install pyrea

Authors: Marcus D. Bloice and Bastian Pfeifer, Medical University of Graz.

Overview
--------
Pyrea allows complex, layered ensembles to be created using an easy to use API.

Formally, ensembles are created as follows. A view :math:`V \in \mathbb{R}^{n \times p}`, where :math:`n` is the number of
samples and :math:`p` is the number of predictors, and is associated with a clustering method, :math:`c`.

An ensemble, :math:`\mathcal{E}`, can be modelled using a set of views :math:`\mathcal{V}` and an associated fusion algorithm,
:math:`f`.

.. math::
   \mathcal{V} \leftarrow \{(V \in \mathbb{R}^{n\times p}, c)\}

.. math::
   \mathcal{E}(\mathcal{V}, f) \rightarrow \widetilde{V}\in \mathbb{R}^{p\times p}

.. math::
   \mathcal{V} \leftarrow \{(\widetilde{V}\in \mathbb{R}^{p\times p}, c)\}

From the above equations we can see that a specified ensemble :math:`\mathcal{E}` creates a view
:math:`\widetilde{V} \in \mathbb{R}^{p\times p}` which again can be used to specify :math:`\mathcal{V}` including an
associated clustering algorithm :math:`c`. With this concept it is possible to *layer-wise* stack views and ensembles to a
arbitrary complex ensemble architecture. It should be noted, however, that the resulting view of a specified ensemble
:math:`\mathcal{E}` reflects an affinity matrix of dimension :math:`p \times p`, and thus only clustering methods which
accepts an affinity or a distance matrix as an input are applicable.

Example "Parea"
~~~~~~~~~~~~~~~
In the paper by Pfeifer et al.\ [1]_, a method called Parea\ :sub:`hc` was introduced. We show here how the Parea workflow
from this paper can be reproduced using Pyrea.

Indeed, the Parea\ :sub:`hc` method supports two different hierarchical ensemble architectures. Parea\ :sub:`hc`\ :sup:`1` clusters multiple data views using two *hierarchical* clustering methods hc\ :sub:`1` and hc\ :sub:`2`. The resulting fused matrices :math:`\widetilde{V}` are clustered with the same methods and the results are combined to a final consensus. A formal description of the Parea\ :sub:`hc`\ :sup:`1` is:

.. math::
   \mathcal{V}_{1} \leftarrow \{(V_{1},hc_{1}),(V_{2},hc_{1}),\ldots, (V_{N},hc_{1})\},
   \quad
   \mathcal{V}_{2} \leftarrow \{(V_{1},hc_{2}),(V_{2},hc_{2}),\ldots, (V_{N},hc_{2})\}


.. math::
   \mathcal{E}_{1}(\mathcal{V}_{1}, f) \rightarrow \widetilde{V}_{1},
   \quad
   \mathcal{E}_{2}(\mathcal{V}_{2}, f) \rightarrow \widetilde{V}_{2}

.. math::
   \mathcal{V}_{3} \leftarrow \{(\widetilde{V}_{1},hc_{1}),(\widetilde{V}_{2},hc_{2})\}

.. math::
   \mathcal{E}_{3}(\mathcal{V}_{3}, f) \rightarrow \widetilde{V}_{3}.

The affinity matrix :math:`\widetilde{V}_{3}` is then clustered with :math:`hc_{1}` and :math:`hc_{2}` from the first layer,
and the consensus of the obtained clustering solutions reflect the final cluster assignments.

Pyrea Implementation
~~~~~~~~~~~~~~~~~~~~
In order to implement the method descrbed in the paper by Pfeifer et al.\ [1]_ and to demonstrate the API with an example,
we provide here the source code to implement this method:

.. code-block:: python
   :caption: Implementing the Parea method using Pyrea
   :linenos:

   import pyrea

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

For complete documentation of all modules, classes, and functions, see the sections below.


Main Documentation
==================

.. toctree::
   :maxdepth: 3

   code
   extending
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==========

.. [1] Pfeifer, Bastian and Schimek, Michael G, "A hierarchical clustering and data fusion approach for disease subtype discovery", *Journal of Biomedical Informatics*, Volume 113 (2021)