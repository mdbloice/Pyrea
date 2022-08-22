Extending Pyrea
===============
Pyrea has been designed to be customisable and extensible. For example Pyrea
can be extended by using, for example, custom clustering algorimths in your
workflows.


Custom Clustering Algorithms
----------------------------
When you create a view, it must have an associated clustering algorithm.

Normally, views are created one of the built in clustering methods, such as
follows:

.. code-block:: python
    :caption: Creating a view
    :linenos:

        import pyrea

        data = [[1,2,3],
                [4,5,6],
                [7,8,9]]

        c = ClusterMethod('ward')

        v = pyrea.View(data, c)

        #or, more simply:
        v = pyrea.View(data, pyrea.ClusterMethod('ward'))

The built in methods derive from SciKit-Learn, and can be determined printing
:code:`pyrea.CLUSTER_METHODS`:

.. code-block:: python
    :caption: Viewing Pyrea's built in clustering algorimths.
    :linenos:

        import pyrea

        print(pyrea.CLUSTER_METHODS)

        # Outputs
        ['ward', 'complete', 'single', 'average', 'random_method']

However, you may wish to supply your own algorithm to your view and within your
workflow.

To do this, create your own `ClusterMethod` type, as follows:

.. code-block:: python
    :caption: Creating a custom clustering algorithm
    :linenos:

        import pyrea

        class CustomClusterMethod(pyrea.ClusterMethod):
            def __init__():
                super().__init__()
                # Your implementation here
                pass

If your implementation is of type :class:`ClusterMethod` then you can do the following:

.. code-block:: python
    :caption: Using a custom ClusterMethod class
    :linenos:

        import pyrea

        # Assuming you have defined your custom class as above:

        data = [[1,2,3],
                [4,5,6],
                [7,8,9]]

        c = CustomClusterMethod()

        v = pyrea.View(data, c)

More details to follow...