import math
from collections import OrderedDict
from typing import Callable, Union, Tuple

import traja
import numpy as np
import pandas as pd
import scipy

from pandas.core.dtypes.common import (
    is_datetime_or_timedelta_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)
from scipy import signal
from scipy.spatial.distance import directed_hausdorff, euclidean

from traja import TrajaDataFrame


def smooth_sg(trj: TrajaDataFrame, w: int = None, p: int = 3):
    """Returns``DataFrame`` of trajectory after Savitzky-Golay filtering.

    Args:
      trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory
      w (int): window size (Default value = None)
      p (int): polynomial order (Default value = 3)

    Returns:
      trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory

    """
    if w is None:
        w = p + 3 - p % 2

    if w % 2 != 1:
        raise Exception(f"Invalid smoothing parameter w ({w}): n must be odd")
    trj.x = signal.savgol_filter(trj.x, window_length=w, polyorder=p, axis=0)
    trj.y = signal.savgol_filter(trj.y, window_length=w, polyorder=p, axis=0)
    trj = fill_in_traj(trj)
    return trj


def angles(trj: TrajaDataFrame, lag: int = 1):
    """Returns angles w.r.t. x-axis."""
    if lag > 1:
        raise NotImplementedError("Lag must be 1.")
    dx = trj.x.diff(lag)
    distance = calc_displacement(trj)
    angles = np.rad2deg(np.arccos(np.abs(dx) / distance))
    # Correction for 360-degree angle range
    angles[angles >= 180] -= 360
    angles[angles < -180] += 360
    return angles


def step_lengths(trj: TrajaDataFrame):
    """Length of the steps of ``trj``.

    Args:
      trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory

    Returns:

    """
    displacement = traja.trajectory.calc_displacement(trj)
    return displacement


def polar_to_z(r: float, theta: float):
    """Converts polar coordinates ``z`` and ``theta`` to complex number ``z``.

    Args:
      r (float): step size
      theta (float): angle

    Returns:

    """
    return r * np.exp(1j * theta)


def cartesian_to_polar(xy: np.ndarray):
    """Convert :class:`numpy.ndarray` ``xy`` to polar coordinates ``r`` and ``theta``.

    Args:
      xy (:class:`numpy.ndarray`): x,y coordinates

    Returns:
        r, theta (tuple of float): step-length and angle

    """
    assert xy.ndim == 2, f"Dimensions are {xy.ndim}, expecting 2"
    x, y = np.split(xy, [-1], axis=1)
    x, y = np.squeeze(x), np.squeeze(y)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return r, theta


def expected_sq_displacement(trj: TrajaDataFrame, n: int = None, eqn1: bool = True):
    """Expected displacement.

    .. note::

        This method is experimental and needs testing.

    """
    # TODO: Fix and test implementation
    sl = step_lengths(trj)
    ta = angles(trj)
    l = np.mean(sl)
    l2 = np.mean(sl ** 2)
    c = np.mean(np.cos(ta))
    s = np.mean(np.sin(ta))
    s2 = s ** 2

    if eqn1:
        # Eqn 1
        alpha = np.arctan2(s, c)
        gamma = ((1 - c) ** 2 - s2) * np.cos((n + 1) * alpha) - 2 * s * (
            1 - c
        ) * np.sin((n + 1) * alpha)
        esd = (
            n * l2
            + 2 * l ** 2 * ((c - c ** 2 - s2) * n - c) / ((1 - c) ** 2 + s2)
            + 2
            * l ** 2
            * ((2 * s2 + (c + s2) ** ((n + 1) / 2)) / ((1 - c) ** 2 + s2) ** 2)
            * gamma
        )
        return abs(esd)
    else:
        # Eqn 2
        esd = n * l2 + 2 * l ** 2 * c / (1 - c) * (n - (1 - c ** n) / (1 - c))
        return esd


def traj_from_coords(
    track, x_col=1, y_col=2, time_col=None, fps=4, spatial_units="m", time_units="s"
):
    """Create TrajaDataFrame from coordinates."""
    if not isinstance(track, traja.TrajaDataFrame):
        if isinstance(track, np.ndarray) and track.shape[1] == 2:
            trj = traja.from_xy(track)
        elif isinstance(track, pd.DataFrame):
            trj = traja.TrajaDataFrame(track)
    else:
        trj = track
    trj.traja.spatial_units = spatial_units
    trj.traja.time_units = time_units

    def rename(col, name, trj):
        if isinstance(col, int):
            trj.rename(columns={col: name})
        else:
            if col not in trj:
                raise Exception(f"Missing column {col}")
            trj.rename(columns={col: name})
        return trj

    # Ensure column names are as expected
    trj = rename(x_col, "x", trj)
    trj = rename(y_col, "y", trj)
    if time_col is not None:
        trj = rename(time_col, "time", trj)

    # Allocate times if they aren't already known
    if "time" not in trj:
        if fps is None:
            raise Exception(
                (
                    "Cannot create a trajectory without times: either fps or a time column must be specified"
                )
            )
        # Assign times to each frame, starting at 0
        trj["time"] = pd.Series(np.arange(0, len(trj)) / fps)

    # Get displacement time for each coordinate, with the first point at time 0
    trj["dt"] = trj.time - trj.time.iloc[0]

    ...
    return trj


def distance(A: traja.TrajaDataFrame, B: traja.TrajaDataFrame, method="dtw"):
    """Returns distance between two trajectories.

    Args:
        A (:class:`~traja.frame.TrajaDataFrame`) : Trajectory 1
        B (:class:`~traja.frame.TrajaDataFrame`) : Trajectory 2
        method (str): ``dtw`` for dynamic time warping, ``hausdorff`` for Hausdorff

    Returns:
        distance (float): Distance
    """
    if method is "hausdorff":
        dist0 = directed_hausdorff(A, B)[0]
        dist1 = directed_hausdorff(B, A)[0]
        symmetric_dist = max(dist0, dist1)
        return symmetric_dist
    elif method is "dtw":
        try:
            from fastdtw import fastdtw
        except ImportError:
            raise ImportError(
                """            
            fastdtw is not installed. Install it with: 
            pip install fastdtw.

            """
            )
        distance, path = fastdtw(A, B, dist=euclidean)
        return distance


def to_shapely(trj):
    """Returns shapely object for area, bounds, etc. functions.

    Args:
        trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory

    Returns:
      shapely.geometry.linestring.LineString -- Shapely shape.

    .. doctest::

        >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
        >>> shape = traja.to_shapely(df)
        >>> shape.is_closed
        False

    """
    from shapely.geometry import shape

    coords = trj[["x", "y"]].values
    tracks_obj = {"type": "LineString", "coordinates": coords}
    tracks_shape = shape(tracks_obj)
    return tracks_shape


def transition_matrix(grid_indices1D: np.ndarray):
    """Returns ``np.ndarray`` of Markov transition probability matrix for grid cell transitions.

    Args:
        grid_indices1D

    Returns:
        M (:class:`numpy.ndarray`)

    """
    n = 1 + max(grid_indices1D.flatten())  # number of states

    M = [[0] * n for _ in range(n)]

    for (i, j) in zip(grid_indices1D, grid_indices1D[1:]):
        M[i][j] += 1

    # Convert to probabilities
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
    return np.array(M)


def _bins_to_tuple(trj, bins: Union[int, Tuple[int, int]] = 10):
    """Returns tuple of x, y bins

    Args:
        trj: Trajectory
        bins: The bin specification:
            If int, the number of bins for the smallest of the two dimensions such that (min(nx,ny)=bins).
            If [int, int], the number of bins in each dimension (nx, ny = bins).

    Returns:
        bins (Sequence[int,int]): Bins (nx, ny)

    """
    if bins is None:
        bins = 10
    if isinstance(bins, int):
        # make aspect equal
        xlim, ylim = _get_xylim(trj)
        aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        if aspect >= 1:
            bins = (bins, int(bins * aspect))
        else:
            bins = (int(bins / aspect), bins)

    assert len(bins) == 2, f"bins should be length 2 but is {len(bins)}"
    return bins


def calculate_flow_angles(grid_indices: np.ndarray):
    """Calculate average flow between grid indices."""

    bins = (grid_indices[:, 0].max(), grid_indices[:, 1].max())
    n = bins[0] * bins[1]  # number of states

    M = np.empty((bins[1], bins[0]), dtype=np.ndarray)

    grid_indices -= 1  # zero-indexing
    for (i, j) in zip(grid_indices, grid_indices[1:]):
        ix = i[0]
        iy = i[1]
        jx = j[0]
        jy = j[1]

        if np.array_equal(i, j):
            angle = None
        elif ix == jx and iy > jy:  # move towards y origin (down by default)
            angle = 3 * np.pi / 2
        elif ix == jx and iy < jy:  # move towards y origin (up by default)
            angle = np.pi / 2
        elif ix < jx and iy == jy:  # move right
            angle = 0
        elif ix > jx and iy == jy:  # move left
            angle = np.pi
        elif ix > jx and iy > jy:  # move towards y origin (top left)
            angle = 3 * np.pi / 4
        elif ix > jx and iy < jy:  # move away from y origin (bottom left)
            angle = 5 * np.pi / 4
        elif ix < jx and iy < jy:  # move away from y origin (bottom right)
            angle = 7 * np.pi / 4
        elif ix < jx and iy > jy:  # move towards y origin (top right)
            angle = np.pi / 4
        if angle is not None:
            M[iy, ix] = np.append(M[iy, ix], angle)

    U = np.ones_like(M)  # x component of arrow
    V = np.empty_like(M)  # y component of arrow
    for i, row in enumerate(M):
        for j, angles in enumerate(row):
            x = y = 0
            average_angle = None
            if angles is not None and len(angles) > 1:
                for angle in angles:
                    if angle is None:
                        continue
                    x += np.cos(angle)
                    y += np.sin(angle)
                # average_angle = np.arctan2(y, x)
                U[i, j] = x
                V[i, j] = y
            else:
                U[i, j] = 0
                V[i, j] = 0

    return U.astype(float), V.astype(float)


def _grid_coords1D(grid_indices: np.ndarray):
    """Convert 2D grid indices to 1D indices."""
    if isinstance(grid_indices, pd.DataFrame):
        grid_indices = grid_indices.values
    grid_indices1D = []
    nr_cols = int(grid_indices[:, 0].max())
    for coord in grid_indices:
        grid_indices1D.append(
            coord[1] * nr_cols + coord[0]
        )  # nr_rows * col_length + nr_cols

    return np.array(grid_indices1D, dtype=int)


def transitions(trj: TrajaDataFrame, **kwargs):
    """Get first-order Markov model for transitions between grid cells."""
    if "xbin" not in trj.columns or "ybin" not in trj.columns:
        grid_indices = grid_coordinates(trj, **kwargs)
    else:
        grid_indices = trj[["xbin", "ybin"]]

    grid_indices1D = _grid_coords1D(grid_indices)
    transitions_matrix = transition_matrix(grid_indices1D)
    return transitions_matrix.astype(int)


def grid_coordinates(
    trj: TrajaDataFrame,
    bins: Union[int, tuple] = None,
    xlim: tuple = None,
    ylim: tuple = None,
    assign: bool = False,
):
    """Returns ``DataFrame`` of trajectory discretized into 2D lattice grid coordinates.
    Args:
        trj (~`traja.frame.TrajaDataFrame`): Trajectory
        bins (tuple or int)
        xlim (tuple)
        ylim (tuple)
        assign (bool): Return updated original dataframe

    Returns:
        trj (~`traja.frame.TrajaDataFrame`): Trajectory

    """
    xmin = trj.x.min() if xlim is None else xlim[0]
    xmax = trj.x.max() if xlim is None else xlim[1]
    ymin = trj.y.min() if ylim is None else ylim[0]
    ymax = trj.y.max() if ylim is None else ylim[1]

    bins = _bins_to_tuple(trj, bins)

    xbins = np.linspace(xmin, xmax, bins[0])
    ybins = np.linspace(ymin, ymax, bins[1])

    xbin = np.digitize(trj.x, xbins)
    ybin = np.digitize(trj.y, ybins)

    if assign:
        trj["xbin"] = xbin
        trj["ybin"] = ybin
        return trj
    return pd.DataFrame({"xbin": xbin, "ybin": ybin}, dtype=int)


def generate(
    n: int = 1000,
    random: bool = True,
    step_length: int = 2,
    angular_error_sd: float = 0.5,
    angular_error_dist: Callable = None,
    linear_error_sd: float = 0.2,
    linear_error_dist: Callable = None,
    fps: float = 50,
    spatial_units: str = "m",
    seed: int = None,
    **kwargs,
):
    """Generates a trajectory.

    If ``random`` is ``True``, the trajectory will
    be a correlated random walk/idiothetic directed walk (Kareiva & Shigesada,
    1983), corresponding to an animal navigating without a compass (Cheung,
    Zhang, Stricker, & Srinivasan, 2008). If ``random`` is ``False``, it
    will be a directed walk/allothetic directed walk/oriented path, corresponding
    to an animal navigating with a compass (Cheung, Zhang, Stricker, &
    Srinivasan, 2007, 2008).

    By default, for both random and directed walks, errors are normally
    distributed, unbiased, and independent of each other, so are **simple
    directed walks** in the terminology of Cheung, Zhang, Stricker, & Srinivasan,
    (2008). This behaviour may be modified by specifying alternative values for
    the ``angular_error_dist`` and/or ``linear_error_dist`` parameters.

    The initial angle (for a random walk) or the intended direction (for a
    directed walk) is ``0`` radians. The starting position is ``(0, 0)``.

    Args:
      n (int):  (Default value = 1000)
      random (bool):  (Default value = True)
      step_length:  (Default value = 2)
      angular_error_sd (float):  (Default value = 0.5)
      angular_error_dist (Callable):  (Default value = None)
      linear_error_sd (float):  (Default value = 0.2)
      linear_error_dist (Callable):  (Default value = None)
      fps (float):  (Default value = 50)
      spatial_units:  (Default value = 'm')
      **kwargs: Additional arguments

    Returns:
        trj (:class:`traja.frame.TrajaDataFrame`): Trajectory

    .. note::

        Based on Jim McLean's `trajr <https://github.com/JimMcL/trajr>`_, ported to Python.

        **Reference**: McLean, D. J., & Skowron Volponi, M. A. (2018). trajr: An R package for characterisation of animal
        trajectories. Ethology, 124(6), 440-448. https://doi.org/10.1111/eth.12739.

    """
    if seed is None:
        np.random.seed(0)
    if angular_error_dist is None:
        angular_error_dist = np.random.normal(loc=0.0, scale=angular_error_sd, size=n)
    if linear_error_dist is None:
        linear_error_dist = np.random.normal(loc=0.0, scale=linear_error_sd, size=n)
    angular_errors = angular_error_dist
    linear_errors = linear_error_dist
    step_lengths = step_length + linear_errors
    # Don't allow negative lengths
    step_lengths[step_lengths < 0] = 0
    steps = polar_to_z(step_lengths, angular_errors)

    if random:
        # Accumulate angular errors
        coords = np.zeros(n + 1, dtype=np.complex)
        angle = 0
        for i in range(n):
            angle += angular_errors[i]
            length = step_length + linear_errors[i]
            coords[i + 1] = coords[i] + polar_to_z(r=length, theta=angle)
    else:
        coords = np.append(complex(0), np.cumsum(steps))

    x = coords.real
    y = coords.imag

    df = traja.TrajaDataFrame(data={"x": x, "y": y})
    if fps in (0, None):
        raise Exception("fps must be greater than 0")
    df.fps = fps
    time = df.index / fps
    df["time"] = time
    df.spatial_units = spatial_units
    for key, value in kwargs.items():
        df.__dict__[key] = value
    # Update metavars
    metavars = dict(angular_error_sd=angular_error_sd, linear_error_sd=linear_error_sd)
    df.__dict__.update(metavars)

    return df


def _resample_time(trj: TrajaDataFrame, step_time: Union[float, int]):
    if not is_datetime_or_timedelta_dtype(trj.index):
        raise Exception(f"{trj.index.dtype} is not datetime or timedelta.")
    return trj.resample(step_time).agg({"x": np.mean, "y": np.mean})


def resample_time(trj: TrajaDataFrame, step_time: str, new_fps: bool = None):
    """Returns a ``TrajaDataFrame`` resampled to consistent `step_time` intervals.

    Args:
        trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory
        step_time (str): step time interval (eg, '1s')
        new_fps (bool, optional): new fps

    Results:
        trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory


    .. doctest::

        >>> from traja import generate
        >>> from traja.trajectory import resample_time
        >>> df = generate()
        >>> resampled = resample_time(df, '2s')
        >>> resampled.head()
              time          x          y
        0 00:00:00  14.555071 -26.482614
        1 00:00:02  -3.582797  -6.491297
        2 00:00:04  -4.299709  26.937443
        3 00:00:06 -25.337042  42.131848
        4 00:00:08  33.069915  32.780830

    """
    time_col = _get_time_col(trj)
    if time_col is "index" and is_datetime64_any_dtype(trj.index):
        _trj = _resample_time(trj, step_time)
    elif time_col is "index" and is_timedelta64_dtype(trj.index):
        _trj = _resample_time(trj, step_time)
    elif time_col:
        if isinstance(step_time, str):
            try:
                if "." in step_time:
                    raise NotImplementedError("Fractional step time not implemented.")
            except Exception:
                raise NotImplementedError(
                    f"Inferring from time format {step_time} not yet implemented."
                )
        _trj = trj.set_index(time_col)
        _trj.index = pd.to_timedelta(_trj.index, unit="s")
        _trj = _resample_time(_trj, step_time)
        _trj.reset_index(inplace=True)
    else:
        raise NotImplementedError(
            f"Time column ({time_col}) not of expected data type."
        )
    return _trj


def rotate(df, angle: Union[float, int] = 0, origin: tuple = None):
    """Returns a ``TrajaDataFrame`` Rotate a trajectory `angle` in radians.

    Args:
        trj (:class:`traja.frame.TrajaDataFrame`): Trajectory
        angle (float): angle in radians
        origin (tuple. optional): rotate around point (x,y)

    Returns:
        trj (:class:`traja.frame.TrajaDataFrame`): Trajectory

    .. note::

        Based on Lyle Scott's `implementation <https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302>`_.

    """
    trj = df.copy()
    # Calculate current orientation
    if isinstance(trj, traja.TrajaDataFrame):
        xy = df.traja.xy
    elif isinstance(trj, pd.DataFrame):
        trj = df[["x", "y"]]

    x, y = np.split(xy, [-1], axis=1)
    if origin is None:
        # Assume middle of x and y is origin
        origin = ((x.max() - x.min()) / 2, (y.max() - y.min()) / 2)

    offset_x, offset_y = origin
    new_coords = []

    for x, y in xy:
        adjusted_x = x - offset_x
        adjusted_y = y - offset_y
        cos_rad = math.cos(angle)
        sin_rad = math.sin(angle)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        new_coords.append((qx, qy))

    new_xy = np.array(new_coords)
    x, y = np.split(new_xy, [-1], axis=1)
    trj["x"] = x
    trj["y"] = y
    return trj


def rediscretize_points(trj: TrajaDataFrame, R: Union[float, int]):
    """Returns a ``TrajaDataFrame`` rediscretized to a constant step length `R`.

    Args:
      trj (:class:`traja.frame.TrajaDataFrame`): Trajectory
      R (float): Rediscretized step length (eg, 0.02)

    Returns:
      rt (:class:`numpy.ndarray`): rediscretized trajectory

    """
    rt = _rediscretize_points(trj, R)

    if len(rt) < 2:
        raise RuntimeError(
            f"Step length {R} is too large for path (path length {len(self._trj)})"
        )
    rt = traja.from_xy(rt)
    return rt


def _rediscretize_points(trj: TrajaDataFrame, R: Union[float, int]):
    """Helper function for :func:`traja.trajectory.rediscretize`.

    Args:
      trj (:class:`traja.frame.TrajaDataFrame`): Trajectory
      R (float): Rediscretized step length (eg, 0.02)

    Returns:
      result (:class:`numpy.ndarray`): Rediscretized coordinates

    """
    # TODO: Implement with complex numbers
    points = trj[["x", "y"]].dropna().values.astype("float64")
    n_points = len(points)
    result = np.empty((128, 2))
    p0 = points[0]
    result[0] = p0
    step_nr = 0
    candidate_start = 1  # running index of candidate

    while candidate_start <= n_points:
        # Find the first point `curr_ind` for which |points[curr_ind] - p_0| >= R
        curr_ind = np.NaN
        for i in range(
            candidate_start, n_points
        ):  # range of search space for next point
            d = np.linalg.norm(points[i] - result[step_nr])
            if d >= R:
                curr_ind = i  # curr_ind is in [candidate, n_points)
                break
        if np.isnan(curr_ind):
            # End of path
            break

        # The next point may lie on the same segment
        candidate_start = curr_ind

        # The next point lies on the segment p[k-1], p[k]
        curr_result_x = result[step_nr][0]
        prev_x = points[curr_ind - 1, 0]
        curr_result_y = result[step_nr][1]
        prev_y = points[curr_ind - 1, 1]

        # a = 1 if points[k, 0] <= xk_1 else 0
        lambda_ = np.arctan2(
            points[curr_ind, 1] - prev_y, points[curr_ind, 0] - prev_x
        )  # angle
        cos_l = np.cos(lambda_)
        sin_l = np.sin(lambda_)
        U = (curr_result_x - prev_x) * cos_l + (curr_result_y - prev_y) * sin_l
        V = (curr_result_y - prev_y) * cos_l - (curr_result_x - prev_x) * sin_l

        # Compute distance H between (X_{i+1}, Y_{i+1}) and (x_{k-1}, y_{k-1})
        H = U + np.sqrt(abs(R ** 2 - V ** 2))
        XIp1 = H * cos_l + prev_x
        YIp1 = H * sin_l + prev_y

        # Increase array size progressively to make the code run (significantly) faster
        if len(result) <= step_nr + 1:
            result = np.concatenate((result, np.empty_like(result)))

        # Save the point
        result[step_nr + 1] = np.array([XIp1, YIp1])
        step_nr += 1

    # Truncate result
    result = result[: step_nr + 1]
    return result


def _has_cols(trj: TrajaDataFrame, cols: list):
    """Check if `trj` has `cols`."""
    return set(cols).issubset(trj.columns)


def calc_turn_angle(trj: TrajaDataFrame):
    """Return a ``Series`` of floats with turn angles.

    Args:
      trj (:class:`traja.frame.TrajaDataFrame`): Trajectory

    Returns:
        turn_angle (:class:`~pandas.Series`): Turn angle

    .. doctest::

        >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
        >>> traja.calc_turn_angle(df)
        0    NaN
        1    NaN
        2    0.0
        Name: turn_angle, dtype: float64

    """
    if "heading" not in trj:
        heading = calc_heading(trj)
    else:
        heading = trj.heading
    turn_angle = heading.diff().rename("turn_angle")
    # Correction for 360-degree angle range
    turn_angle.loc[turn_angle >= 180] -= 360
    turn_angle.loc[turn_angle < -180] += 360
    return turn_angle


def calc_angle(trj: TrajaDataFrame):
    """Returns a ``Series`` with angle between steps as a function of displacement w.r.t x axis.

    Args:
       trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory

    Returns:
      angle (:class:`pandas.Series`): Angle series.

    """
    if not _has_cols(trj, ["dx", "displacement"]):
        displacement = calc_displacement(trj)
    else:
        displacement = trj.displacement

    angle = np.rad2deg(np.arccos(np.abs(trj.x.diff()) / displacement))
    return angle


def calc_displacement(trj: TrajaDataFrame):
    """Returns a ``Series`` of ``float`` displacement between consecutive indices.

    Args:
        trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory

    Returns:
      displacement (:class:`pandas.Series`): Displacement series.

    .. doctest::

        >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
        >>> traja.calc_displacement(df)
        0         NaN
        1    1.414214
        2    1.414214
        Name: displacement, dtype: float64

    """
    displacement = np.sqrt(
        np.power(trj.x.shift() - trj.x, 2) + np.power(trj.y.shift() - trj.y, 2)
    )
    displacement.name = "displacement"
    return displacement


def calc_derivatives(trj: TrajaDataFrame):
    """Returns derivatives ``displacement`` and ``displacement_time`` as DataFrame.

    Args:
      trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory

    Returns:
      derivs (:class:`~pandas.DataFrame`): Derivatives.

    .. doctest::

        >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3],'time':[0., 0.2, 0.4]})
        >>> traja.calc_derivatives(df)
           displacement  displacement_time
        0           NaN                0.0
        1      1.414214                0.2
        2      1.414214                0.4

    """

    time_col = _get_time_col(trj)
    if time_col is None:
        raise Exception("Missing time information in trajectory.")

    if not "displacement" in trj:
        displacement = calc_displacement(trj)
    else:
        displacement = trj.displacement

    # get cumulative seconds
    if is_datetime64_any_dtype(trj[time_col]):
        displacement_time = (
            trj[time_col].astype(int).div(10 ** 9).diff().fillna(0).cumsum()
        )
    else:
        displacement_time = trj[time_col].diff().fillna(0).cumsum()

    # TODO: Create DataFrame directly
    derivs = pd.DataFrame(
        OrderedDict(displacement=displacement, displacement_time=displacement_time)
    )

    return derivs


def calc_heading(trj: TrajaDataFrame):
    """Calculate trajectory heading.

    Args:
      assign (bool): (Default value = True)

    Returns:
        heading (:class:`pandas.Series`): heading as a ``Series``

    ..doctest::

        >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
        >>> traja.calc_heading(df)
        0     NaN
        1    45.0
        2    45.0
        Name: heading, dtype: float64

    """
    if not _has_cols(trj, ["angle"]):
        angle = calc_angle(trj)
    else:
        angle = trj.angle

    dx = trj.x.diff()
    dy = trj.y.diff()
    # Get heading from angle
    mask = (dx > 0) & (dy >= 0)
    trj.loc[mask, "heading"] = angle[mask]
    mask = (dx >= 0) & (dy < 0)
    trj.loc[mask, "heading"] = -angle[mask]
    mask = (dx < 0) & (dy <= 0)
    trj.loc[mask, "heading"] = -(180 - angle[mask])
    mask = (dx <= 0) & (dy > 0)
    trj.loc[mask, "heading"] = 180 - angle[mask]
    return trj.heading


def speed_intervals(
    trj: TrajaDataFrame,
    faster_than: float = None,
    slower_than: float = None,
    interpolate_times: bool = True,
):
    """Calculate speed time intervals.

    Returns a dictionary of time intervals where speed is slower and/or faster than specified values.

    Args:
      faster_than (float, optional): Minimum speed threshold. (Default value = None)
      slower_than (float or int, optional): Maximum speed threshold. (Default value = None)
      interpolate_times (bool, optional): Interpolate times between steps. (Default value = True)

    Returns:
      result (:class:`~collections.OrderedDict`) -- time intervals as dictionary.

    .. note::

        Implementation ported to Python, heavily inspired by Jim McLean's trajr package.

    """
    derivs = get_derivatives(trj)

    if faster_than is not None:
        pass
    if slower_than is not None:
        pass

    # Calculate trajectory speeds
    speed = derivs["speed"]
    times = derivs["speed_times"]
    flags = np.full(len(speed), 1)

    if faster_than is not None:
        flags = flags & (speed > faster_than)
    if slower_than is not None:
        flags = flags & (speed < slower_than)

    changes = np.diff(flags)
    stop_frames = np.where(changes == -1)[0]
    start_frames = np.where(changes == 1)[0]

    # Handle situation where interval begins or ends outside of trajectory
    if len(start_frames) > 0 or len(stop_frames) > 0:
        # Assume interval started at beginning of trajectory, since we don't know what happened before that
        if len(stop_frames) > 0 and (
            len(start_frames) == 0 or stop_frames[0] < start_frames[0]
        ):
            start_frames = np.append(1, start_frames)
        # Similarly, assume that interval can't extend past end of trajectory
        if (
            len(stop_frames) == 0
            or start_frames[len(start_frames) - 1] > stop_frames[len(stop_frames) - 1]
        ):
            stop_frames = np.append(stop_frames, len(speed))

    stop_times = times[stop_frames]
    start_times = times[start_frames]

    if interpolate_times and len(start_frames) > 0:
        # TODO: Implement
        raise NotImplementedError()
        r = linear_interp_times(
            slower_than, faster_than, speed, times, start_frames, start_times
        )
        start_times = r[:, 0]
        stop_times = r[:, 1]

    durations = stop_times - start_times
    result = traja.TrajaDataFrame(
        OrderedDict(
            start_frame=start_frames,
            start_time=start_times,
            stop_frame=stop_frames,
            stop_time=stop_times,
            duration=durations,
        )
    )
    return result


def get_derivatives(trj: TrajaDataFrame):
    """Returns derivatives ``displacement``, ``displacement_time``, ``speed``, ``speed_times``, ``acceleration``,
    ``acceleration_times`` as dictionary.

    Args:
        trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory

    Returns:
      derivs (:class:`~pd.DataFrame`) : Derivatives

    .. doctest::

        >> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3],'time':[0.,0.2,0.4]})
        >> df.traja.get_derivatives() #doctest: +SKIP
           displacement  displacement_time     speed  speed_times  acceleration  acceleration_times
        0           NaN                0.0       NaN          NaN           NaN                 NaN
        1      1.414214                0.2  7.071068          0.2           NaN                 NaN
        2      1.414214                0.4  7.071068          0.4           0.0                 0.4

    """
    if not _has_cols(trj, ["displacement", "displacement_time"]):
        derivs = calc_derivatives(trj)
        d = derivs["displacement"]
        t = derivs["displacement_time"]
    else:
        d = trj.displacement
        t = trj.displacement_time
        derivs = OrderedDict(displacement=d, displacement_time=t)
    if is_datetime_or_timedelta_dtype(t):
        # Convert to float divisible series
        # TODO: Add support for other time units
        t = t.dt.total_seconds()
    v = d[1 : len(d)] / t.diff()
    v.rename("speed")
    vt = t[1 : len(t)].rename("speed_times")
    # Calculate linear acceleration
    a = v.diff() / vt.diff().rename("acceleration")
    at = vt[1 : len(vt)].rename("accleration_times")

    data = dict(speed=v, speed_times=vt, acceleration=a, acceleration_times=at)
    derivs = derivs.merge(pd.DataFrame(data), left_index=True, right_index=True)
    return derivs


def _get_xylim(trj: TrajaDataFrame) -> Tuple[Tuple, Tuple]:
    if (
        "xlim" in trj.__dict__
        and "ylim" in trj.__dict__
        and isinstance(trj.xlim, tuple)
    ):
        return trj.xlim, trj.ylim
    else:
        xlim = trj.x.min(), trj.x.max()
        ylim = trj.y.min(), trj.y.max()
        return xlim, ylim


def coords_to_flow(trj: TrajaDataFrame, bins: Union[int, tuple] = None):
    """Calculate grid cell flow from trajectory.

    Args:
        trj (trajectory)
        bins (int or tuple)

    Returns:
        X (:class:`~numpy.ndarray`): X coordinates of arrow locations
        Y (:class:`~numpy.ndarray`): Y coordinates of arrow locations
        U (:class:`~numpy.ndarray`): X component of vector data
        V (:class:`~numpy.ndarray`): Y component of vector data

    """
    xlim, ylim = _get_xylim(trj)
    bins = _bins_to_tuple(trj, bins)

    X, Y = np.meshgrid(
        np.linspace(trj.x.min(), trj.x.max(), bins[0]),
        np.linspace(trj.y.min(), trj.y.max(), bins[1]),
    )

    if "xbin" not in trj.columns or "ybin" not in trj.columns:
        grid_indices = traja.grid_coordinates(trj, bins=bins, xlim=xlim, ylim=ylim)
    else:
        grid_indices = trj[["xbin", "ybin"]]

    U, V = traja.calculate_flow_angles(grid_indices.values)

    return X, Y, U, V


def from_xy(xy: np.ndarray):
    """Convenience function for initializing :class:`~traja.frame.TrajaDataFrame` with x,y coordinates.

    Args:
        xy (:class:`numpy.ndarray`): x,y coordinates

    Returns:
        traj_df (:class:`~traja.frame.TrajaDataFrame`): Trajectory as dataframe

    .. doctest::

        >>> import numpy as np
        >>> xy = np.array([[0,1],[1,2],[2,3]])
        >>> traja.from_xy(xy)
           x  y
        0  0  1
        1  1  2
        2  2  3

    """
    df = traja.TrajaDataFrame.from_records(xy, columns=["x", "y"])
    return df


def fill_in_traj(trj: TrajaDataFrame):
    # FIXME: Implement
    return trj


def _get_time_col(trj: TrajaDataFrame):
    # Check if saved in metadata
    time_col = trj.__dict__.get("time_col", None)
    if time_col:
        return time_col
    # Check if index is datetime
    if is_datetime_or_timedelta_dtype(trj.index):
        return "index"
    # Check if any column contains 'time'
    time_cols = [col for col in trj if "time" in col.lower()]
    if time_cols:
        # Try first column
        time_col = time_cols[0]
        if is_datetime_or_timedelta_dtype(trj[time_col]):
            return time_col
        else:
            # Time column is float, etc. but not datetime64.
            # FIXME: Add conditional return, etc.
            return time_col
    else:
        # No time column found
        return None
