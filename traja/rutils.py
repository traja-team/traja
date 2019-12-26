try:
    import rpy2
except ImportError:
    raise ImportError(
        "Python package rpy2 is required for this. Install it with"
        "\n"
        "pip install rpy2"
        ""
    )
import rpy2.robjects.packages as rpackages

try:
    import rpy2.robjects.pandas2ri as rpandas
except ModuleNotFoundError as e:
    if "tzlocal" in e.msg:
        raise ModuleNotFoundError(
            e.msg + "\n Install tzlocal with `pip install tzlocal`."
        )
    else:
        raise ModuleNotFoundError(e)
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

__all__ = ["import_adehabitat", "import_trajr", "plot_ltraj", "to_trajr", "to_ltraj"]


rpandas.activate()

installed_rpackages = []
ADEHABITAT_INSTALLED = False
TRAJR_INSTALLED = False


def import_adehabitat(suppress_messages=True):
    global installed_rpackages
    if not 'adehabitat' in installed_rpackages:
        utils = rpackages.importr("utils", suppress_messages=suppress_messages)
        print("Importing adehabitat")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages("adehabitatLT")
        installed_rpackages.append('adehabitat')
    adehabitat = importr("adehabitatLT", suppress_messages=suppress_messages)
    return adehabitat


def import_trajr(suppress_messages=True):
    global installed_rpackages
    if not 'trajr' in installed_rpackages:
        utils = rpackages.importr("utils", suppress_messages=suppress_messages)
        print("Importing trajr")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages("trajr")
        installed_rpackages.append('trajr')
    trajr = importr("trajr")
    return trajr

def import_moveHMM(suppress_messages=True):
    global installed_rpackages
    if not 'moveHMM' in installed_rpackages:
        utils = rpackages.importr("utils", suppress_messages=suppress_messages)
        print("Importing moveHMM")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages("moveHMM")
        installed_rpackages.append('moveHMM')
    moveHMM = importr("moveHMM")
    return moveHMM


def plot_ltraj(ltraj, id=1):
    """Plot `ltraj` using R method."""
    adehabitat = import_adehabitat()
    adehabitat.plot_ltraj(ltraj, id=id)


def to_trajr(trj):
    """Convert trajectory to R `trajr` object. Default fps is 30.

    Args:
        trajectory (:class:`~traja.main.TrajaDataFrame`): trajectory


    Returns:
        traj (:class:`rpy2.robjects.vectors.DataFrame`): column names are ['x', 'y', 'time', 'displacementTime',
                                                            'polar', 'displacement']

    .. doctest::

        >>> import traja
        >>> df = traja.TrajaDataFrame({'x':range(5),'y':range(5)})
        >>> trjr_df = traja.rutils.to_trajr(df) # doctest: +SKIP
        >>> [x for x in trjr_df.names] # doctest: +SKIP
        ...
        ['x', 'y', 'id', 'time', 'displacementTime', 'polar', 'displacement']


    """
    from traja.trajectory import _get_time_col

    trajr = import_trajr()
    if "id" not in trj.__dict__.keys():
        trj["id"] = 0
    time_col = _get_time_col(trj)
    if time_col == "index":
        trj["time"] = trj.index
        time_col = "time"
    fps = trj.fps
    spatial_units = trj.spatial_units or "m"
    time_units = trj.time_units or "s"

    trj_rdf = rpandas.py2rpy(trj)

    trajr_trj = trajr.TrajFromCoords(
        trj_rdf,
        xCol="x",
        yCol="y",
        timeCol=time_col or rpy2.rinterface.NULL,
        fps=fps or 30,
        spatialUnits=spatial_units,
        timeUnits=time_units,
    )

    return trajr_trj


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

        >>> import traja
        >>> df = traja.TrajaDataFrame({'x':range(5),'y':range(5)})
        >>> ltraj = traja.rutils.to_ltraj(df) # doctest: +SKIP
        >>> print(ltraj[0]) # doctest: +SKIP
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

    df = trj[["x", "y"]]

    if typeII:
        raise NotImplementedError(
            "datetime series not yet implemented for this method."
        )
        # FIXME: Add date converted from rpy2.robjects.POSIXct
        # date = None
        # ltraj = adehabitat.as_ltraj(df, id=id, date=date, typeII=True)  # Doesn't work
    else:
        ltraj = adehabitat.as_ltraj(df, id=id, typeII=False)[0]
    return ltraj

def prep_moveData(trj, id='ID', type="UTM", coord_names=("x","y")):
    """Convert trajectory to R moveHMM package moveData format."""
    moveHMM = import_moveHMM()
    df = trj[['x','y']]
    moveData = moveHMM.prep_data(df, type=type, coordNames = coord_names)
    return moveData

def plot_moveData(moveData, compact=True):
    """Plot animal tracks, time series and step lengths and turn distributions with
    the R package moveHMM.""" 
    moveHMM = import_moveHMM()
    return moveHMM.plot(prepped_trj, compact=compact)

def fitHMM(moveData, covariates=[], standardize=True, nb_states=2):
    moveHMM = import(moveHMM)
    if standardize:
        mean = ro.r['mean']
        std = ro.r['std']
        for cov in covariates:
            moveData[cov] = mean(moveData[cov])[0] / std(moveData[cov])[0]
    mu0 = ro.FloatVector((0.1,1)) # step mean (two parameters: one for each state)
    sigma0 = ro.FloatVector((0.1,1)) # step SD
    zeromass0 = ro.FloatVector((0.1,0.05)) # step zero-mass
    stepPar0 = ro.FloatVector((mu0,sigma0,zeromass0))
    pi = ro.r['pi'][0]
    angleMean0 = ro.FloatVector((pi,0)) # angle mean
    kappa0 = ro.IntVector((1,1)) # angle concentration
    anglePar0 = ro.FloatVector((angleMean0,kappa0))
    ## call to fitting function
    model = moveHMM.fitHMM(data=moveData,
        nbStates=nb_states,
        stepPar0=stepPar0,
        anglePar0=anglePar0,
        # formula=~dist_water,  # TODO
        )
    return model

def plot_hmm_model(fitHMM_model):
    return model.plot()