.. ipython:: python :okwarning:
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

See the :ref:`gallery<sphx_glr_gallery>` for more examples.

.. automodule:: traja.plotting
    :members: bar_plot, plot, plot_quiver, plot_contour, plot_surface, plot_stream, plot_flow, plot_actogram, polar_bar

Trip Grid
---------

Trip grid can be plotted for :class:`~traja.frame.TrajaDataFrame`s with :func:`~traja.accessor.TrajaAccessor.trip_grid`:

.. ipython:: python :okwarning:

    import traja
    from traja import trip_grid

    df = traja.TrajaDataFrame({'x':range(10),'y':range(10)})
    @savefig trip_grid.png
    hist, image = trip_grid(df);


If only the histogram is need for further computation, use the `hist_only` option:

.. ipython:: python

    hist, _ = trip_grid(df, hist_only=True)
    print(hist[:5])


Highly dense plots be more easily visualized using the `bins` and `log` argument:

.. ipython:: python :okwarning:

    # Generate random walk
    df = traja.generate(1000)

    @savefig trip_grid_log.png
    trip_grid(df, bins=32, log=True);

The plot can also be normalized into a density function with `normalize`:

.. ipython:: python :okwarning:

    @savefig trip_grid_normalized.png
    hist, _ = trip_grid(df, normalize=True);


Animate
-------

.. autofunction:: traja.plotting.animate