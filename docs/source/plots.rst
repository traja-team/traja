.. ipython:: python
   :suppress:

   import matplotlib
   import pandas as pd
   from traja.plotting import trip_grid
   orig = matplotlib.rcParams['figure.figsize']
   matplotlib.rcParams['figure.figsize'] = [orig[0] * 1.5, orig[1]]
   import matplotlib.pyplot as plt
   plt.close('all')


Plotting Paths
==============

Making plots of trajectories is easy using the :meth:`~traja.accessor.TrajaAccessor.plot` method.

Trip Grid
---------

Trip grid can be plotted for :class:`~traja.trajadataframe.TrajaDataFrame`s with :func:`~traja.accessor.TrajaAccessor.trip_grid`:

.. ipython:: python :okwarning:

    import traja

    df = traja.TrajaDataFrame({'x':range(10),'y':range(10)})
    @savefig trip_grid.png
    hist, image = df.traja.trip_grid();


If only the histogram is need for further computation, use the `hist_only` option:

.. ipython:: python

    from traja.plotting import trip_grid

    hist, _ = trip_grid(df, hist_only=True)
    print(hist[:5)


Highly dense plots be more easily visualized using the `bins` and `log` argument:

.. ipython:: python :okwarning:

    from traja import generate

    df = generate(1000)

    @savefig trip_grid_log.png
    df.traja.trip_grid(bins=32, log=True);

The plot can also be normalized into a density function with `normalize`:

.. ipython:: python :okwarning:

    @savefig trip_grid_normalized.png
    hist, _ = trip_grid(df, normalize=True);

Plotting Grid Cell Flow
=======================

Trajectories can be discretized into grid cells and the average flow from
each grid cell to its neighbor can be plotted with :func:`traja.plotting.plot_flow`:

:func:`~traja.plotting.plot_flow` `kind` Arguments
--------------------------------------------------

* `surface` - 3D surface plot extending :meth:`mpl_toolkits.mplot3D.Axes3D.plot_surface``
* `contourf` - Filled contour plot extending :meth:`matplotlib.axes.Axes.contourf`
* `quiver` - Quiver plot extending :meth:`matplotlib.axes.Axes.quiver`
* `stream` - Stream plot extending :meth:`matplotlib.axes.Axes.streamplot`