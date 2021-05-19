Smoothing and Analysis
======================

Smoothing
---------

Smoothing can be performed using :func:`~traja.trajectory.smooth_sg`.

.. autofunction:: traja.trajectory.smooth_sg

.. ipython::

    df = traja.generate()
    smoothed = traja.smooth_sg(df, w=101)
    smoothed.traja.plot()

.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/images/smoothed.png


Length
------

Length of trajectory can be calculated using :func:`~traja.trajectory.length`.

.. autofunction:: traja.trajectory.length

Distance
--------

Net displacement of trajectory (start to end) can be calculated using :func:`~traja.trajectory.distance`.

.. autofunction:: traja.trajectory.distance

Displacement
------------

Displacement (distance travelled) can be calculated using :func:`~traja.trajectory.calc_displacement`.

.. autofunction:: traja.trajectory.calc_displacement

Derivatives
-----------

.. autofunction:: traja.trajectory.get_derivatives

Speed Intervals
---------------

.. autofunction:: traja.trajectory.speed_intervals