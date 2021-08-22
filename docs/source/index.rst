.. traja documentation master file, created by
   sphinx-quickstart on Mon Jan 28 23:36:32 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

traja |version|
===============

Trajectory Analysis in Python

Traja allows analyzing trajectory datasets using a wide range of tools, including pandas and R.
Traja extends the capability of pandas :class:`~pandas.DataFrame` specific for animal or object trajectory
analysis in 2D, and provides convenient interfaces to other geometric analysis packages (eg, shapely).

Description
-----------

The Traja Python package is a toolkit for the numerical characterization and analysis
of the trajectories of moving animals. Trajectory analysis is applicable in fields as
diverse as optimal foraging theory, migration, and behavioural mimicry
(e.g. for verifying similarities in locomotion).
A trajectory is simply a record of the path followed by a moving object.
Traja operates on trajectories in the form of a series of locations (as x, y coordinates) with times.
Trajectories may be obtained by any method which provides this information,
including manual tracking, radio telemetry, GPS tracking, and motion tracking from videos.

The goal of this package (and this document) is to aid biological researchers, who may not have extensive
experience with Python, to analyse trajectories
without being restricted by a limited knowledge of Python or programming.
However, a basic understanding of Python is useful.

If you use Traja in your publications, please cite:

.. code-block:: txt

   @article{Shenk2021,
   doi = {10.21105/joss.03202},
   url = {https://doi.org/10.21105/joss.03202},
   year = {2021},
   publisher = {The Open Journal},
   volume = {6},
   number = {63},
   pages = {3202},
   author = {Justin Shenk and Wolf Byttner and Saranraj Nambusubramaniyan and Alexander Zoeller},
   title = {Traja: A Python toolbox for animal trajectory analysis},
   journal = {Journal of Open Source Software}
   }

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Installation <install>
   Examples Gallery <gallery/index>

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   Reading and Writing Files <reading>
   Pandas Indexing and Resampling <pandas>
   Generate Random Walk <generate>
   Smoothing and Analysis <calculations>
   Turns <turns>
   Plotting Paths <plots>
   Periodicity <periodicity>
   Plotting Grid Cell Flow <grid_cell>
   Rediscretizing Trajectories <rediscretize>
   Clustering and Dimensionality Reduction <clustering>
   Collections / Scenes <collections>
   Predicting Trajectories <predictions>

.. toctree::
   :maxdepth: 1
   :caption: Reference Guide

   Reference to All Attributes and Methods <reference>
   Bugs and Support <support>

.. toctree::
   :maxdepth: 1
   :caption: Developer

   Contributing to Traja <contributing>  


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
