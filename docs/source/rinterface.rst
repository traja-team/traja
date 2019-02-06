R interface
===========

`ryp2` allows connecting R packages to Python. Install rpy2 with ``pip install rpy2``.

.. code-block:: python

    import traja
    from traja import rutils

    df = traja.generate()

Convert objects to adehabitat class ltraj for further analysis with R
=====================================================================

`adehabitat <https://www.rdocumentation.org/packages/adehabitat/versions/1.8.20>`_
is a widely used R library for animal tracking and trajectory
analysis.

.. code-block:: python

    ltraj = rutils.to_ltraj(df)
    rutils.plot_ltraj(ltraj)

.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/source/_static/ltraj_plot.png

Perform further analysis in Python
==================================
Data frame is stored in first index.

.. code-block:: python

    print(ltraj[0].head())

