Reference
=============

Accessor Methods
----------------

The following methods are available via :class:`traja.accessor.TrajaAccessor`:

.. automodule:: traja.accessor
    :members:
    :undoc-members:
    :noindex:

Plotting functions
------------------

The following methods are available via :mod:`traja.plotting`:

.. automethod:: traja.plotting.animate

.. automethod:: traja.plotting.bar_plot

.. automethod:: traja.plotting.color_dark

.. automethod:: traja.plotting.fill_ci

.. automethod:: traja.plotting.find_runs

.. automethod:: traja.plotting.plot

.. automethod:: traja.plotting.plot_3d

.. automethod:: traja.plotting.plot_actogram

.. automethod:: traja.plotting.plot_autocorrelation

.. automethod:: traja.plotting.plot_contour

.. automethod:: traja.plotting.plot_clustermap

.. automethod:: traja.plotting.plot_flow

.. automethod:: traja.plotting.plot_quiver

.. automethod:: traja.plotting.plot_stream

.. automethod:: traja.plotting.plot_surface

.. automethod:: traja.plotting.plot_transition_matrix

.. automethod:: traja.plotting.plot_xy

.. automethod:: traja.plotting.polar_bar

.. automethod:: traja.plotting.plot_prediction

.. automethod:: traja.plotting.sans_serif

.. automethod:: traja.plotting.stylize_axes

.. automethod:: traja.plotting.trip_grid


Analysis
--------

The following methods are available via :mod:`traja.trajectory`:

.. automethod:: traja.trajectory.angles

.. automethod:: traja.trajectory.calc_angle

.. automethod:: traja.trajectory.calc_convex_hull

.. automethod:: traja.trajectory.calc_derivatives

.. automethod:: traja.trajectory.calc_displacement

.. automethod:: traja.trajectory.calc_heading

.. automethod:: traja.trajectory.calc_turn_angle

.. automethod:: traja.trajectory.calc_flow_angles

.. automethod:: traja.trajectory.cartesian_to_polar

.. automethod:: traja.trajectory.coords_to_flow

.. automethod:: traja.trajectory.determine_colinearity

.. automethod:: traja.trajectory.distance_between

.. automethod:: traja.trajectory.distance

.. automethod:: traja.trajectory.euclidean

.. automethod:: traja.trajectory.expected_sq_displacement

.. automethod:: traja.trajectory.fill_in_traj

.. automethod:: traja.trajectory.from_xy

.. automethod:: traja.trajectory.generate

.. automethod:: traja.trajectory.get_derivatives

.. automethod:: traja.trajectory.grid_coordinates

.. automethod:: traja.trajectory.inside

.. automethod:: traja.trajectory.length

.. automethod:: traja.trajectory.polar_to_z

.. automethod:: traja.trajectory.rediscretize_points

.. automethod:: traja.trajectory.resample_time

.. automethod:: traja.trajectory.return_angle_to_point

.. automethod:: traja.trajectory.rotate

.. automethod:: traja.trajectory.smooth_sg

.. automethod:: traja.trajectory.speed_intervals

.. automethod:: traja.trajectory.step_lengths

.. automethod:: traja.trajectory.to_shapely

.. automethod:: traja.trajectory.traj_from_coords

.. automethod:: traja.trajectory.transition_matrix

.. automethod:: traja.trajectory.transitions

io functions
------------

The following methods are available via :mod:`traja.parsers`:

.. automethod:: traja.parsers.read_file

.. automethod:: traja.parsers.from_df


TrajaDataFrame
--------------

A ``TrajaDataFrame`` is a tabular data structure that contains ``x``, ``y``, and ``time`` columns.

All pandas ``DataFrame`` methods are also available, although they may
not operate in a meaningful way on the ``x``, ``y``, and ``time`` columns.

Inheritance diagram:

.. inheritance-diagram:: traja.TrajaDataFrame

TrajaCollection
---------------

A ``TrajaCollection`` holds multiple trajectories for analyzing and comparing trajectories.
It has limited accessibility to lower-level methods.

.. autoclass:: traja.frame.TrajaCollection

.. automethod:: traja.frame.TrajaCollection.apply_all

.. automethod:: traja.frame.TrajaCollection.plot


API Pages
---------

.. currentmodule:: traja
.. autosummary::
  :template: autosummary.rst
  :toctree: reference/

  TrajaDataFrame
  TrajaCollection
  read_file
