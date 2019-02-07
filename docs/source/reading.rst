Reading and Writing Files
=========================

Reading trajectory data
-----------------------

traja allows reading files via :func:`traja.read_file`.

.. code-block:: python

    # trajectory.csv
    # x,y
    # 1,1
    # 1,2
    # 1,3

    import traja

    df = traja.read_file('trajectory.csv')

This returns a `TrajaDataFrame` with access to all pandas and traja methods.

Any keyword arguments passed to `read_file` will be passed to :meth:`pandas.read_csv`.