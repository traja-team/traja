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