Reading and Writing Files
=========================

Reading trajectory data
-----------------------

traja allows reading files via :func:`traja.io.read_file`. For example a CSV file ``trajectory.csv`` with the
following contents::


    x,y
    1,1
    1,2
    1,3

Could be read in like:

.. code-block:: python

    import traja

    df = traja.read_file('trajectory.csv')

``read_file`` returns a `TrajaDataFrame` with access to all pandas and traja methods.

.. automodule:: traja.accessor
   .. automethod::

Any keyword arguments passed to `read_file` will be passed to :meth:`pandas.read_csv`.

Writing trajectory data
-----------------------

Files can be saved using the built-in pandas :func:`pandas.to_csv`.

.. code-block:: python

    df.to_csv('trajectory.csv')
