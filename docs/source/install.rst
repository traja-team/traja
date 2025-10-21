Installation
============

Installing traja
----------------

traja requires Python 3.8+ to be installed. For installing on Windows,
it is recommend to download and install via conda_.

To install via conda::

    conda install -c conda-forge traja

To install via pip::

   pip install traja

To install the latest development version, clone the `GitHub` repository and use the setup script::

   git clone https://github.com/traja-team/traja.git
   cd traja
   pip install .

Dependencies
------------

Installation with pip should also include all dependencies, but a complete list is

- numpy_
- matplotlib_
- scipy_
- pandas_

To install all optional dependencies run::

  pip install 'traja[all]'


.. _GitHub: https://github.com/justinshenk/github

.. _numpy: http://www.numpy.org

.. _pandas: http://pandas.pydata.org

.. _scipy: https://docs.scipy.org/doc/scipy/reference/

.. _shapely: http://toblerity.github.io/shapely

.. _matplotlib: http://matplotlib.org

.. _conda: https://docs.conda.io/en/latest/