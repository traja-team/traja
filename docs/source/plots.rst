Making Plots
============

Making plots of trajectories is easy using the :meth:`~traja.accessor.TrajaAccessor.plot` method.

Trip Grid
=========

Trip grid can be plotted for `TrajaDataFrames` with :meth:`~traja.utils.trip_grid`::

    df.traja.trip_grid()

or for any pandas `DataFrame` containing `x` and `y` columns with::

    from traja.utils import trip_grid

    df = pd.DataFrame({'x':range(10),'y':range(10)})
    hist, image = trip_grid(df)

If only the histogram is need for further computation, use the `hist_only` option::

    hist, _ = trip_grid(df, hist_only=True)

The plot can also be normalized into a density function with `normalize`::

    hist, _ = trip_grid(df, normalize=True)


Highly dense plots be more easily visualized using the `bins` and `log` argument::

    from traja.utils import generate

    df = generate(1000)
    df.traja.trip_grid(bins=30, log=True)