Reference
=============

Accessor Methods
----------------

The following methods are available via :class:`traja.accessor.TrajaAccessor`:

.. automodule:: traja.accessor
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Plotting functions
------------------

The following methods are available via :mod:`traja.plotting`:

.. automodule:: traja.plotting
    :members:
    :undoc-members:
    :show-inheritance:

R functions
------------------

The following methods are available via :mod:`traja.rutils`:

.. automodule:: traja.rutils
    :members:
    :undoc-members:
    :show-inheritance:

Trajectory functions
--------------------

The following methods are available via :mod:`traja.trajectory`:

.. automodule:: traja.trajectory
    :members:
    :undoc-members:
    :show-inheritance:

io functions
------------

The following methods are available via :mod:`traja.parsers`:

.. automodule:: traja.parsers
    :members:
    :undoc-members:
    :show-inheritance:

TrajaDataFrame
--------------

A ``TrajaDataFrame`` is a tabular data structure that contains ``x``, ``y``, and ``time`` columns.

All pandas ``DataFrame`` methods are also available, although they may
not operate in a meaningful way on the ``x``, ``y``, and ``time`` columns.

Inheritance diagram:

.. inheritance-diagram:: traja.TrajaDataFrame


