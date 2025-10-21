Traja |Python-ver| |Travis| |PyPI| |Conda| |RTD| |Gitter| |Black| |License| |Binder| |Codecov| |DOI| |JOSS|
===========================================================================================================

|Colab|

.. |Python-ver| image:: https://img.shields.io/badge/python-3.8+-blue.svg
    :target: https://www.python.org/downloads/release/python-380/
    :alt: Python 3.8+

.. |Travis| image:: https://travis-ci.org/traja-team/traja.svg?branch=master
    :target: https://travis-ci.org/traja-team/traja

.. |PyPI| image:: https://badge.fury.io/py/traja.svg
    :target: https://badge.fury.io/py/traja

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/traja.svg
    :target: https://anaconda.org/conda-forge/traja

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

.. |Binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/justinshenk/traja/master?filepath=demo.ipynb

.. |Codecov| image:: https://codecov.io/gh/traja-team/traja/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/traja-team/traja

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5069231.svg
   :target: https://doi.org/10.5281/zenodo.5069231

.. |Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/justinshenk/traja/blob/master/demo.ipynb
   
.. |JOSS| image:: https://joss.theoj.org/papers/0f25dc08671e0ec54714f09597d116cb/status.svg
   :target: https://joss.theoj.org/papers/0f25dc08671e0ec54714f09597d116cb

Traja is a Python library for trajectory analysis. It extends the capability of
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
analyze trajectories without being restricted by a limited knowledge of
Python or programming. However, a basic understanding of Python is
useful.

If you use traja in your publications, please cite the repo 

.. code-block::

    @software{justin_shenk_2019_3237827,
      author       = {Justin Shenk and
                      the Traja development team},
      title        = {justinshenk/traja},
      month        = jun,
      year         = 2019,
      publisher    = {Zenodo},
      version      = {latest},
      doi          = {10.5281/zenodo.3237827},
      url          = {https://doi.org/10.5281/zenodo.3237827}
    }


Installation and setup
----------------------

To install traja with conda, run

``conda install -c conda-forge traja``

or with pip

``pip install traja``.

Import traja into your Python script or via the Python command-line with
``import traja``.

Trajectories with traja
-----------------------

Traja stores trajectories in pandas DataFrames, allowing any pandas
functions to be used.

Load trajectory with x, y and time coordinates:

.. code-block:: python

    import traja

    df = traja.read_file('coords.csv')

Once a DataFrame is loaded, use the ``.traja`` accessor to access the
visualization and analysis methods:

.. code-block:: python

    df.traja.plot(title='Cage trajectory')


Analyze Trajectory
------------------

.. csv-table:: The following functions are available via ``traja.trajectory.[method]``
   :header: "Function", "Description"
   :widths: 30, 80
   
   "``calc_derivatives``", "Calculate derivatives of x, y values "
   "``calc_turn_angles``", "Calculate turn angles with regard to x-axis "
   "``transitions``", "Calculate first-order Markov model for transitions between grid bins"
   "``generate``", "Generate random walk"
   "``resample_time``", "Resample to consistent step_time intervals"
   "``rediscretize_points``", "Rediscretize points to given step length"
   
For up-to-date documentation, see `https://traja.readthedocs.io <https://traja.readthedocs.io>`_.

Deep Learning Integration
--------------------------

Traja provides production-ready features for training neural networks on trajectory data:

**Data Augmentation** - Create training variations for robust models:

.. code-block:: python

    # Rotation, noise, scaling, reversal, subsampling
    rotated = df.traja.augment_rotate(angle=45)
    noisy = df.traja.augment_noise(sigma=0.1)
    scaled = df.traja.augment_scale(factor=1.5)

**Sequence Processing** - Standardize trajectory lengths for batching:

.. code-block:: python

    # Pad or truncate to fixed length
    padded = df.traja.pad_trajectory(target_length=200, mode='edge')
    truncated = df.traja.truncate_trajectory(target_length=100, mode='random')
    normalized = df.traja.normalize_trajectory()

**Feature Extraction** - Generate ML-ready features:

.. code-block:: python

    # Extract displacement, speed, turn_angle, heading, acceleration
    features = df.traja.extract_features()

**PyTorch Integration** - Seamless tensor conversion:

.. code-block:: python

    tensor = df.traja.to_tensor()  # Convert to PyTorch tensor

**Dataset Utilities** - Train/val/test splitting:

.. code-block:: python

    trajectories = [traja.generate(n=100) for _ in range(50)]
    train, val, test = traja.trajectory.train_test_split(
        trajectories, train_size=0.7, val_size=0.15, test_size=0.15
    )

**3D Support** - All features work with x, y, z coordinates:

.. code-block:: python

    df_3d = traja.TrajaDataFrame({'x': x, 'y': y, 'z': z})
    tensor_3d = df_3d.traja.to_tensor()  # Shape: (n_points, 3)

**GPS/Lat-Long Support** - Work with GPS coordinates:

.. code-block:: python

    traj = traja.from_latlon(lat, lon)  # Convert GPS to local x,y

**Visualization Enhancements** - Better trajectory analysis and exploration:

.. code-block:: python

    # Interactive plots with plotly
    fig = df.traja.plot_interactive()  # Zoom, pan, rotate

    # Heatmap showing time spent in locations
    df.traja.plot_heatmap(bins=50)

    # Speed and acceleration profiles
    df.traja.plot_speed()
    df.traja.plot_acceleration()

    # Comprehensive 4-panel analysis
    df.traja.plot_trajectory_components()

**Performance Optimization** - Fast parallel processing:

.. code-block:: python

    # Process 100 trajectories in parallel
    trajectories = [traja.generate(n=1000) for _ in range(100)]
    results = traja.trajectory.batch_process(
        trajectories,
        lambda t: t.traja.normalize_trajectory(),
        n_jobs=-1  # Use all CPUs
    )

See the `Deep Learning documentation <https://traja.readthedocs.io/en/latest/deep_learning.html>`_ and ``examples/deep_learning_demo.ipynb`` for complete examples.

Random walk
-----------

Generate random walks with

.. code-block:: python

    df = traja.generate(n=1000, step_length=2)
    df.traja.plot()

.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/source/_static/walk_screenshot.png
   :alt: walk\_screenshot.png


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
