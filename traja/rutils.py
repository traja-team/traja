#! /usr/local/env python3
try:
    import rpy2
except ImportError:
    raise ImportError("Python package rpy2 is required for this. Install it with"
                      ""
                      "pip install rpy2"
                      "")
import rpy2.robjects.packages as rpackages
import pandas as pd
import rpy2.robjects.pandas2ri as rpandas
from rpy2.robjects.packages import importr

rpandas.activate()

ADEHABITAT_INSTALLED=False

def import_adehabitat():
    global ADEHABITAT_INSTALLED
    if not ADEHABITAT_INSTALLED:
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages('adehabitatLT')
        ADEHABITAT_INSTALLED = True
    adehabitat = importr('adehabitatLT')
    return adehabitat

def plot_ltraj(ltraj, id=1):
    """Plot `ltraj` using R method."""
    adehabitat = import_adehabitat()
    adehabitat.plot_ltraj(ltraj, id=1)


def to_ltraj(trj, id=1, typeII=False):
    """Convert trajectory to R `ltraj` object.

    Args:
        trajectory (:class:`~traja.main.TrajaDataFrame`): trajectory
        id (str, optional): id of animal/target
        typeII (bool):  contains datatime format time series (not yet supported)

    Returns:
        ltraj (:class:`rpy2.robjects.vectors.ListVector`): first index (ie, ltraj[0]) is pandas DataFrame including
                                                            columns 'x', 'y', 'date', 'dx', 'dy', 'dist', 'dt',
                                                            'R2n', 'abs.angle', 'rel.angle'

    .. doctest::

        >>> import traja.rutils as rutils # doctest:+ELLIPSIS
        >>>
        >>> df=traja.TrajaDataFrame({'x':range(5),'y':range(5)})
        >>> ltraj = rutils.to_ltraj(df)
        >>> print(ltraj[0])
        ...
           x  y  date          dx    ...              dt   R2n  abs.angle  rel.angle
        0  0  0     1           1    ...               1   0.0   0.785398        NaN
        1  1  1     2           1    ...               1   2.0   0.785398        0.0
        2  2  2     3           1    ...               1   8.0   0.785398        0.0
        3  3  3     4           1    ...               1  18.0   0.785398        0.0
        4  4  4     5 -2147483648    ...     -2147483648  32.0        NaN        NaN

        [5 rows x 10 columns]

    """
    adehabitat = import_adehabitat()

    df = trj[['x','y']]

    if typeII:
        raise NotImplementedError("datetime series not yet implemented for this method.")
        # FIXME: Add date converted from rpy2.robjects.POSIXct
        # date = None
        ltraj = adehabitat.as_ltraj(df, id=id, date=date, typeII=True) # Doesn't work
    else:
        ltraj = adehabitat.as_ltraj(df, id=id, typeII=False)
    return ltraj
