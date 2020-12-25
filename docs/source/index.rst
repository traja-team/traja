.. traja documentation master file, created by
   sphinx-quickstart on Mon Jan 28 23:36:32 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

traja |version|
===============

Trajectory Analysis in Python

traja extends the capability of pandas DataFrame specific for animal trajectory analysis in 2D, and provides convenient interfaces to other geometric analysis packages (eg, shapely).

Description
-----------

The traja Python package is a toolkit for the numerical characterisation and analysis of the trajectories of moving animals. Trajectory analysis is applicable in fields as diverse as optimal foraging theory, migration, and behavioural mimicry (e.g. for verifying similarities in locomotion). A trajectory is simply a record of the path followed by a moving animal. Trajr operates on trajectories in the form of a series of locations (as x, y coordinates) with times. Trajectories may be obtained by any method which provides this information, including manual tracking, radio telemetry, GPS tracking, and motion tracking from videos.

The goal of this package (and this document) is to aid biological researchers, who may not have extensive experience with Python, to analyse trajectories without being handicapped by a limited knowledge of Python or programming. However, a basic understanding of Python is useful.

If you use traja in your publications, please cite it.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Installation <install>
   Examples Gallery <gallery/index>

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   Extending pandas <pandas>
   Making Plots <plots>
   Rediscretizing Trajectory <rediscretize>

.. toctree::
   :maxdepth: 1
   :caption: Reference Guide

   Reference to All Attributes and Methods <reference>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
