Plotting Grid Cell Flow
=======================

Trajectories can be discretized into grid cells and the average flow from
each grid cell to its neighbor can be plotted with :func:`~traja.plotting.plot_flow`, eg:

.. code-block:: python

    traja.plot_flow(df, kind='stream')

:func:`~traja.plotting.plot_flow` ``kind`` Arguments
----------------------------------------------------

* `surface` - 3D surface plot extending :meth:`mpl_toolkits.mplot3D.Axes3D.plot_surface``
* `contourf` - Filled contour plot extending :meth:`matplotlib.axes.Axes.contourf`
* `quiver` - Quiver plot extending :meth:`matplotlib.axes.Axes.quiver`
* `stream` - Stream plot extending :meth:`matplotlib.axes.Axes.streamplot`

See the :ref:`gallery<sphx_glr_gallery>` for more examples.

3D Surface Plot
---------------

.. autofunction:: traja.plotting.plot_surface

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_average_direction_001.png
   :alt: 3D plot

Quiver Plot
-----------

.. autofunction:: traja.plotting.plot_quiver

.. code-block:: python

    traja.plot_quiver(df, bins=32)

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_average_direction_002.png
   :alt: quiver plot

Contour Plot
------------

.. autofunction:: traja.plotting.plot_contour

.. code-block:: python

    traja.plot_contour(df, filled=False, quiver=False, bins=32)

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_average_direction_003.png
   :alt: contour plot

Contour Plot (Filled)
---------------------

.. code-block:: python

    traja.plot_contour(df, filled=False, quiver=False, bins=32)

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_average_direction_004.png
   :alt: contour plot filled

Stream Plot
-----------

.. autofunction:: traja.plotting.plot_stream

.. code-block:: python

    traja.plot_contour(df, bins=32, contourfplot_kws={'cmap':'coolwarm'})

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_average_direction_005.png
   :alt: streamplot
