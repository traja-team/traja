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

