R interface
===========

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

Perform further analysis in Python
==================================
Data frame is stored in first index.

.. code-block:: python

    print(ltraj[0].head())

