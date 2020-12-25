traja
=====

Trajectory Analysis in Python

traja extends the capability of pandas DataFrame specific for animal
trajectory analysis in 2D, and provides convenient interfaces to other
geometric analysis packages (eg, shapely).

Introduction
------------

The traja Python package is a toolkit for the numerical characterisation
and analysis of the trajectories of moving animals. Trajectory analysis
is applicable in fields as diverse as optimal foraging theory,
migration, and behavioural mimicry (e.g. for verifying similarities in
locomotion). A trajectory is simply a record of the path followed by a
moving animal. Trajr operates on trajectories in the form of a series of
locations (as x, y coordinates) with times. Trajectories may be obtained
by any method which provides this information, including manual
tracking, radio telemetry, GPS tracking, and motion tracking from
videos.

The goal of this package (and this document) is to aid biological
researchers, who may not have extensive experience with Python, to
analyse trajectories without being handicapped by a limited knowledge of
Python or programming. However, a basic understanding of Python is
useful.

If you use traja in your publications, please cite [add citation].

Installation and setup
----------------------

To install traja onto your system, run

``pip install traja``

or download the zip file and run the graphical user interface [coming
soon].

Import traja into your Python script or via the Python command-line with
``import traja``.

Trajectories with traja
-----------------------

traja stores trajectories in pandas DataFrames, allowing any pandas
functions to be used.

Load trajectory with x,y and time coordinates:

.. code:: python

    import traja

    df = traja.read_file('coords.csv')

Once a DataFrame is loaded, use the ``.traja`` accessor to access the
visualization and analysis methods:

.. code:: python

    df.traja.plot(title='Cage trajectory')

.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/source/_static//dvc_screenshot.png
   :alt: dvc\_screenshot


Random walk
-----------

Generate random walks with

.. code:: python

    df = traja.generate(n=1000, step_length=2)
    df.traja.plot()

.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/source/_static/walk_screenshot.png
   :alt: walk\_screenshot.png

Demo
----

Coming soon.

Acknowledgements
----------------

traja code implementation and analytical methods (particularly
``rediscretize_points``) are heavily inspired by Jim McLean's R package
`trajr <https://github.com/JimMcL/trajr>`__. Many thanks to Jim for his
feedback.
