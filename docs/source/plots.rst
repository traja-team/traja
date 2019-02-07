.. ipython:: python
   :suppress:

   import matplotlib
   import pandas as pd
   orig = matplotlib.rcParams['figure.figsize']
   matplotlib.rcParams['figure.figsize'] = [orig[0] * 1.5, orig[1]]
   import matplotlib.pyplot as plt
   plt.close('all')


Making Plots
============

Making plots of trajectories is easy using the :meth:`~traja.accessor.TrajaAccessor.plot` method.

Trip Grid
=========

Trip grid can be plotted for :class:`~traja.trajadataframe.TrajaDataFrame`s with :func:`~traja.plotting.trip_grid`:

.. ipython:: python

    import traja
    import matplotlib.pyplot as plt

    df = traja.TrajaDataFrame({'x':range(10),'y':range(10)})
    df.traja.trip_grid()

or for any pandas :class:`~pandas.DataFrame` containing `x` and `y` columns with:

.. ipython:: python

    from traja import trip_grid

    df = pd.DataFrame({'x':range(10),'y':range(10)})
    hist, image = trip_grid(df)

If only the histogram is need for further computation, use the `hist_only` option:

.. ipython:: python

    hist, _ = trip_grid(df, hist_only=True)

The plot can also be normalized into a density function with `normalize`:

.. ipython:: python

    hist, _ = trip_grid(df, normalize=True)

Highly dense plots be more easily visualized using the `bins` and `log` argument:

.. ipython:: python

    from traja import generate

    df = generate(1000)
    df.traja.trip_grid(bins=30, log=True)

.. image:: .. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/source/_static/trip_grid.png