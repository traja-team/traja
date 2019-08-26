Trajectory Collections
======================

TrajaCollection
-------------------

When handling multiple trajectories, Traja allows plotting and analysis simultaneously.

Initialize a :func:`~traja.frame.TrajaCollection` with a dictionary or ``DataFrame`` and ``id_col``.

Initializing with Dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The keys of the dictionary can be used to identify types of objects in the scene, eg, "bus", "car", "person"::

    dfs = {"car0":df0, "car1":df1, "bus0: df2, "person0": df3}


Or, arbitrary numbers can be used to initialize

.. autoclass:: traja.frame.TrajaCollection

.. ipython::

    from traja import TrajaCollection

    dfs = {idx: traja.generate(idx, seed=idx) for idx in range(10,13)}
    trjs = TrajaCollection(dfs)

    print(trjs)

Initializing with a DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A dataframe containing an id column can be passed directly to :func:`~traja.frame.TrajaCollection`, as long as the ``id_col`` is specified::

    trjs = TrajaCollection(df, id_col="id")

Grouped Operations
------------------

Operations can be applied to each trajectory with :func:`~traja.frame.TrajaCollection.apply_all`.

.. automethod:: traja.frame.TrajaCollection.apply_all

Plottting Multiple Trajectories
-------------------------------

Plotting multiple trajectories can be achieved with :func:`~traja.frame.TrajaCollection.plot`.

.. automethod:: traja.frame.TrajaCollection.plot

Colors can be specified for ids by supplying ``colors`` with a lookup dictionary:

.. ipython::

    colors = ["10":"red",
              "11":"red",
              "12":"red",
              "13":"orange",
              "14":"orange"]

or with a substring lookup:

    colors = ["car":"red",
              "bus":"orange",
              "12":"red",
              "13":"orange",
              "14":"orange"]


.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/images/collection_plot.png




