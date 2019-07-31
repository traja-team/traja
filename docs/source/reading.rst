Reading and Writing Files
=========================

Reading trajectory data
-----------------------

traja allows reading files via :func:`traja.parsers.read_file`. For example a CSV file ``trajectory.csv`` with the
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

Data frames can also be read with pandas :func:`pandas.read_csv` and then converted to TrajaDataFrames
with:

.. code-block:: python

    import traja
    import pandas as pd

    df = pd.read_csv('data.csv')

    # If x and y columns are named different than "x" and "y", rename them, eg:
    df = df.rename(columns={"x_col": "x", "y_col": "y"}) # original column names x_col, y_col
    
    # If the time column doesn't include "time" in the name, similarly rename it to "time"

    trj = traja.TrajaDataFrame(df)



Writing trajectory data
-----------------------

Files can be saved using the built-in pandas :func:`pandas.to_csv`.

.. code-block:: python

    df.to_csv('trajectory.csv')
