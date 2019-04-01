traja |Python-ver| |Travis| |PyPI| |RTD| |Gitter| |Black| |License|
=====

.. |Python-ver| image:: https://img.shields.io/badge/python-3.6+-blue.svg
    :target: https://www.python.org/downloads/release/python-360/
    :alt: Python 3.6+

.. |Travis| image:: https://travis-ci.org/justinshenk/traja.svg?branch=master
    :target: https://travis-ci.org/justinshenk/traja

.. |PyPI| image:: https://badge.fury.io/py/traja.svg
    :target: https://badge.fury.io/py/traja

.. |Gitter| image:: https://badges.gitter.im/traja-chat/community.svg
    :target: https://gitter.im/traja-chat/community

.. |RTD| image:: https://readthedocs.org/projects/traja/badge/?version=latest
    :target: https://traja.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black

.. |License| image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License: MIT

traja is a Python library for trajectory analysis. It extends the capability of
pandas DataFrame specific for animal trajectory analysis in 2D, and provides
convenient interfaces to other geometric analysis packages (eg, R and shapely).

Introduction
------------

The traja Python package is a toolkit for the numerical characterization
and analysis of the trajectories of moving animals. Trajectory analysis
is applicable in fields as diverse as optimal foraging theory,
migration, and behavioral mimicry (e.g. for verifying similarities in
locomotion). A trajectory is simply a record of the path followed by a
moving animal. Traja operates on trajectories in the form of a series of
locations (as x, y coordinates) with times. Trajectories may be obtained
by any method which provides this information, including manual
tracking, radio telemetry, GPS tracking, and motion tracking from
videos.

The goal of this package (and this document) is to aid biological
researchers, who may not have extensive experience with Python, to
analyze trajectories without being handicapped by a limited knowledge of
Python or programming. However, a basic understanding of Python is
useful.

If you use traja in your publications, please cite [add citation].

Installation and setup
----------------------

To install traja with conda, run

``conda install -c conda-forge traja``

or with pip

``pip install traja``.

Run the graphical user interface with ``python traja-gui.py``.

Import traja into your Python script or via the Python command-line with
``import traja``.

Trajectories with traja
-----------------------

traja stores trajectories in pandas DataFrames, allowing any pandas
functions to be used.

Load trajectory with x,y and time coordinates:

.. code-block:: python

    import traja

    df = traja.read_file('coords.csv')

Once a DataFrame is loaded, use the ``.traja`` accessor to access the
visualization and analysis methods:

.. code-block:: python

    df.traja.plot(title='Cage trajectory')


Analyze Trajectory
------------------

The following functions are available via ``traja.trajectory.[method]``:

| Function | Description |
|-|-|
| ``calc_derivatives`` | Calculate derivatives of x, y values |
|-|-|
| ``calc_turn_angles`` | Calculate turn angles w.r.t. x-axis |
|-|-|
| ``transitions`` | Calculate first-order Markov model for transitions between grid bins |
|-|-|
| ``generate`` | Generate random walk |
|-|-|
| ``resample_time`` | Resample to consistent ``step_time`` intervals |
|-|-|
| ``rediscretize_points`` | Rediscretize points to given step length |


Random walk
-----------

Generate random walks with

.. code-block:: python

    df = traja.generate(n=1000, step_length=2)
    df.traja.plot()

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_with_traja_003.png
   :alt: walk\_screenshot.png

Rediscretize
------------
Rediscretize the trajectory into consistent step lengths with ``traja.trajectory.rediscretize`` where the ``R`` parameter is
the new step length.

.. code-block:: python

    rt = df.traja.rediscretize(R=5000)
    rt.traja.plot()

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_with_traja_004.png
   :alt: rediscretized


Resample time
-------------
``traja.trajectory.resample_time`` allows resampling trajectories by a ``step_time``.


Flow Plotting
-------------

.. code-block:: python

    df = traja.generate()
    traja.plot_surface(df)

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_average_direction_001.png
   :alt: 3D plot

.. code-block:: python

    traja.plot_quiver(df, bins=32)

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_average_direction_002.png
   :alt: quiver plot

.. code-block:: python

    traja.plot_contour(df, filled=False, quiver=False, bins=32)

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_average_direction_003.png
   :alt: contour plot

.. code-block:: python

    traja.plot_contour(df, filled=False, quiver=False, bins=32)

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_average_direction_004.png
   :alt: contour plot filled

.. code-block:: python

    traja.plot_contour(df, bins=32, contourfplot_kws={'cmap':'coolwarm'})

.. image:: https://traja.readthedocs.io/en/latest/_images/sphx_glr_plot_average_direction_005.png
   :alt: streamplot

Acknowledgements
----------------

traja code implementation and analytical methods (particularly
``rediscretize_points``) are heavily inspired by Jim McLean's R package
`trajr <https://github.com/JimMcL/trajr>`__. Many thanks to Jim for his
feedback.
