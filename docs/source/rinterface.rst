R interface
===========

`rpy2` allows connecting R packages to Python. Install rpy2 with ``pip install rpy2`` or ``conda install -c r rpy2``.

.. code-block:: python

    import traja
    from traja import rutils

    df = traja.generate()

Convert to adehabitat class `ltraj` for further analysis with R
---------------------------------------------------------------

`adehabitat <https://www.rdocumentation.org/packages/adehabitat/versions/1.8.20>`_
is a widely used R library for animal tracking and trajectory
analysis.

.. code-block:: python

    ltraj = rutils.to_ltraj(df)
    rutils.plot_ltraj(ltraj)

.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/source/_static/ltraj_plot.png

Convert to `trajr` object
-------------------------

`trajr <https://cran.r-project.org/web/packages/trajr/vignettes/trajr-vignette.html>`_ is another widely used R package.
Convert a `TrajaDataFrame` to `trajr` dataframe with:

.. code-block:: python

    # Convert to trajr trajectory
    traj = rutils.to_trajr(df)

    # Plot trajectory using trajr backend
    traj.plot_Trajectory(traj)


Perform further analysis in Python
----------------------------------
Data frame is stored in first index.

.. code-block:: python

    print(ltraj.head())

Out::

               x        y date           dx           dy     dist dt       R2n
    0  0.0000000 0.000000    1  1.341849037  1.629900330 2.111193  1  0.000000
    1  1.3418490 1.629900    2  1.022740115  1.923497375 2.178495  1  4.457134
    2  2.3645892 3.553398    3 -0.001297666  1.915536596 1.915537  1 18.217917
    3  2.3632915 5.468934    4 -1.820040071  0.878444059 2.020943  1 35.494389
    4  0.5432514 6.347378    5 -1.810551702 -0.952063825 2.045611  1 40.584334
    5 -1.2673003 5.395315    6 -2.040275034  0.009247058 2.040296  1 30.715469
       abs.angle  rel.angle
    0  0.8820262         NA
    1  1.0821048  0.2000786
    2  1.5714738  0.4893690
    3  2.6919204  1.1204466
    4 -2.6574859  0.9337790
    5  3.1370604 -0.4886389
