Resampling Trajectories
=======================

Rediscretize
------------
Rediscretize the trajectory into consistent step lengths with :meth:`~traja.trajectory.rediscretize` where the `R` parameter is
the new step length.

.. note::

    Based on the appendix in Bovet and Benhamou, (1988) and Jim McLean's
    `trajr <https://github.com/JimMcL/trajr>`_ implementation.


Resample time
-------------
:meth:`~traja.trajectory.resample_time` allows resampling trajectories by a ``step_time``.

`step_time` should be expressed as a number-time unit combination, eg `"2S" for 2 seconds and "2100L" for 2100 milliseconds.

For milliseconds/microseconds/nanoseconds use:

- L - milliseconds
- U - microseconds
- N - nanoseconds

For example:

.. ipython:: python :okwarning:

    import traja

    # Generate a random walk
    df = traja.generate(n=1000) # Time is in 0.02-second intervals
    df.head()

.. ipython:: python :okwarning:

    resampled = traja.resample_time(df, "50L") # 50 milliseconds
    resampled.head()

    fig = resampled.traja.plot()

.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/images/resampled.png


Ramer–Douglas–Peucker algorithm
-------------------------------

.. note::

    Graciously yanked from Fabian Hirschmann's PyPI package ``rdp``.

:func:`~traja.contrib.rdp` reduces the number of points in a line using the Ramer–Douglas–Peucker algorithm::

    from traja.contrib import rdp

    # Create dataframe of 1000 x, y coordinates
    df = traja.generate(n=1000)

    # Extract xy coordinates
    xy = df.traja.xy

    # Reduce points with epsilon between 0 and 1:
    xy_ = rdp(xy, epsilon=0.8)


    len(xy_)

    Output:
    319

Plotting, we can now see the many fewer points are needed to cover a similar area.::

    df = traja.from_xy(xy_)
    df.traja.plot()

.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/source/_static/after_rdp.png

