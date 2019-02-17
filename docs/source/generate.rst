Generate Random Walk
====================

Random walks can be generated using :func:`~traja.trajectory.generate`.

.. code-block:: python

    df = generate(n=1000, fps=30)
    df.traja.plot()

.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/source/_static/walk_screenshot.png

.. note ::

    Based on Jim McLean's `trajr <https://github.com/JimMcL/trajr>`_, ported to Python by Justin Shenk.

    **Reference**: McLean, D. J., & Skowron Volponi, M. A. (2018). trajr: An R package for characterisation of animal
    trajectories. Ethology, 124(6), 440-448. https://doi.org/10.1111/eth.12739.

TODO: `generate` arguments:

**seed**

Specify the random seed number (``int``) with ``seed``.

**

