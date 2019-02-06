#! /usr/local/env python3
import math
from typing import Callable

import traja
import numpy as np
import pandas as pd
import scipy

from traja import TrajaDataFrame
from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype, is_datetime64_any_dtype, is_timedelta64_dtype
from scipy.spatial.distance import directed_hausdorff, euclidean


def smooth_sg(trj, w=None, p=3):
    """Savitzky-Golay filtering.

    Args:
      trj (:class:`~traja.main.TrajaDataFrame`): Trajectory
      w (int): window size (Default value = None)
      p (int): polynomial order (Default value = 3)

    Returns:
      trj: :class:`~traja.main.TrajaDataFrame`

    """
    if w is None:
        w = p + 3 - p % 2

    if (w % 2 != 1):
        raise Exception(f"Invalid smoothing parameter w ({w}): n must be odd")
    trj.x = scipy.signal.savgol_filter(trj.x, window_length=w, polyorder=p, axis=0)
    trj.y = scipy.signal.savgol_filter(trj.y, window_length=w, polyorder=p, axis=0)
    trj = fill_in_traj(trj)
    return trj


def angles(trj, lag=1, compass_direction=None):
    trj['angle'] = np.rad2deg(np.arccos(np.abs(trj['dx']) / trj['distance']))
    # Get heading from angle
    mask = (trj['dx'] > 0) & (trj['dy'] >= 0)
    trj.loc[mask, 'heading'] = trj['angle'][mask]
    mask = (trj['dx'] >= 0) & (trj['dy'] < 0)
    trj.loc[mask, 'heading'] = -trj['angle'][mask]
    mask = (trj['dx'] < 0) & (trj['dy'] <= 0)
    trj.loc[mask, 'heading'] = -(180 - trj['angle'][mask])
    mask = (trj['dx'] <= 0) & (trj['dy'] > 0)
    trj.loc[mask, 'heading'] = (180 - trj['angle'])[mask]
    trj['turn_angle'] = trj['heading'].diff()
    # Correction for 360-degree angle range
    trj.loc[trj.turn_angle >= 180, 'turn_angle'] -= 360
    trj.loc[trj.turn_angle < -180, 'turn_angle'] += 360


def step_lengths(trj):
    """Length of the steps of `trj`.

    Args:
      trj (:class:`~traja.main.TrajaDataFrame`): Trajectory

    Returns:

    """
    raise NotImplementedError()


def polar_to_z(r: float, theta: float):
    """Converts polar coordinates `z` and `theta` to complex number `z`.

    Args:
      r (float): step size
      theta (float): angle

    Returns:

    """
    return r * np.exp(1j * theta)


def cartesian_to_polar(xy):
    """Convert :class:`numpy.ndarray` `xy` to polar coordinates `r` and `theta`.

    Args:
      xy (:class:`numpy.ndarray`): x,y coordinates

    Returns:
        r, theta (tuple of float): step-length and angle

    """
    assert xy.ndim == 2, f"Dimensions are {xy.ndim}, expecting 2"
    x, y = np.split(xy, [-1], axis=1)
    x, y = np.squeeze(x), np.squeeze(y)
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r, theta


def expected_sq_displacement(trj, n=None, eqn1=True, compass_direction=None):
    """Expected displacment.

    .. note::

        This method is experimental and needs testing.

    """
    # TODO: Fix and test implementation
    sl = step_lengths(trj)
    ta = angles(trj, compass_direction=compass_direction)
    l = np.mean(sl)
    l2 = np.mean(sl ^ 2)
    c = np.mean(np.cos(ta))
    s = np.mean(np.sin(ta))
    s2 = s ^ 2

    if eqn1:
        # Eqn 1
        alpha = np.arctan2(s, c)
        gamma = ((1 - c) ^ 2 - s2) * np.cos((n + 1) * alpha) - 2 * s * (1 - c) * np.sin((n + 1) * alpha)
        esd = n * l2 + 2 * l ^ 2 * ((c - c ^ 2 - s2) * n - c) / ((1 - c) ^ 2 + s2) + 2 * l ^ 2 * (
                (2 * s2 + (c + s2) ^ ((n + 1) / 2)) / ((1 - c) ^ 2 + s2) ^ 2) * gamma
        return abs(esd)
    else:
        # Eqn 2
        esd = n * l2 + 2 * l ^ 2 * c / (1 - c) * (n - (1 - c ^ n) / (1 - c))
        return esd


def traj_from_coords(track, x_col=1, y_col=2, time_col=None, fps=4, spatial_units='m', time_units='s'):
    # TODO: Convert to DataFrame if not already
    trj = track
    trj.traja.spatial_units = spatial_units
    trj.traja.time_units = time_units

    def rename(col, name):
        global trj
        if isinstance(col, int):
            trj.rename(columns={col: name})
        else:
            if col not in trj:
                raise Exception(f"Missing column {col}")
            trj.rename(columns={col: name})

    # Ensure column names are as expected
    rename(x_col, 'x')
    rename(y_col, 'y')
    if time_col is not None:
        rename(time_col, 'time')

    # Allocate times if they aren't already known
    if 'time' not in trj:
        if fps is None:
            raise Exception(("Cannot create a trajectory without times: either fps or a time column must be specified"))
        # Assign times to each frame, starting at 0
        trj['time'] = pd.Series(np.arange(0, len(trj) - 1) / fps)

    # Get displacement time for each coordinate, with the first point at time 0
    trj['dt'] = trj.time - trj.time.iloc[0]

    ...
    return trj


# TODO: Delete if unusable
# def traj(filepath, xlim=None, ylim=None, **kwargs):
#     df_test = pd.read_csv(filepath, nrows=100)
#     # Select first col with 'time_stamp' in name as index
#     time_stamp_cols = [x for x in df_test.columns if 'time_stamp' in x]
#     index_col = kwargs.pop('index_col', time_stamp_cols[0])
#
#     df = pd.read_csv(filepath,
#                      date_parser=kwargs.pop('date_parser',
#                                             lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f')),
#                      infer_datetime_format=kwargs.pop('infer_datetime_format', True),
#                      parse_dates=kwargs.pop('parse_dates', True),
#                      index_col=index_col,
#                      **kwargs)
#     if xlim is not None and isinstance(xlim, tuple):
#         df.traja.xlim = xlim
#     if ylim is not None and isinstance(ylim, tuple):
#         df.traja.ylim = ylim
#     return df


def distance(A: traja.TrajaDataFrame, B: traja.TrajaDataFrame, method='dtw'):
    """Calculate distance between two trajectories.

    Args:
        A (:class:`~traja.main.TrajaDataFrame`) : Trajectory 1
        B (:class:`~traja.main.TrajaDataFrame`) : Trajectory 2
        method (str): `dtw` for dynamic time warping, `hausdorff` for Hausdorff

    Returns:
        distance (str): Distance
    """
    if method is 'hausdorff':
        dist0 = directed_hausdorff(A, B)[0]
        dist1 = directed_hausdorff(B, A)[0]
        symmetric_dist = max(dist0, dist1)
        return symmetric_dist
    elif method is 'dtw':
        try:
            from fastdtw import fastdtw
        except ImportError:
            raise ImportError("""            
            fastdtw is not installed. Install it with: 
            pip install fastdtw.

            """)
        distance, path = fastdtw(A, B, dist=euclidean)
        return distance


def generate(n: int = 1000, random: bool = True, step_length: int = 2,
             angular_error_sd: float = 0.5,
             angular_error_dist: Callable = None,
             linear_error_sd: float = 0.2,
             linear_error_dist: Callable = None,
             fps: float = 50,
             spatial_units: str = 'm',
             seed: int = None,
             **kwargs):
    """Generates a trajectory.

    If `random` is `True`, the trajectory will
    be a correlated random walk/idiothetic directed walk (Kareiva & Shigesada,
    1983), corresponding to an animal navigating without a compass (Cheung,
    Zhang, Stricker, & Srinivasan, 2008). If `random` is `False`, it
    will be a directed walk/allothetic directed walk/oriented path, corresponding
    to an animal navigating with a compass (Cheung, Zhang, Stricker, &
    Srinivasan, 2007, 2008).

    By default, for both random and directed walks, errors are normally
    distributed, unbiased, and independent of each other, so are **simple
    directed walks** in the terminology of Cheung, Zhang, Stricker, & Srinivasan,
    (2008). This behaviour may be modified by specifying alternative values for
    the `angular_error_dist` and/or `linear_error_dist` parameters.

    The initial angle (for a random walk) or the intended direction (for a
    directed walk) is `0` radians. The starting position is `(0, 0)`.

    Args:
      n:  (Default value = 1000)
      random:  (Default value = True)
      step_length:  (Default value = 2)
      angular_error_sd:  (Default value = 0.5)
      angular_error_dist:  (Default value = None)
      linear_error_sd:  (Default value = 0.2)
      linear_error_dist:  (Default value = None)
      fps:  (Default value = 50)
      spatial_units:  (Default value = 'm')
      **kwargs:

    Returns:
        trj (:class:`traja.main.TrajaDataFrame`): Trajectory

    .. note::

        Based on Jim McLean's `trajr <https://github.com/JimMcL/trajr>`_, ported to Python by Justin Shenk.

        **Reference**: McLean, D. J., & Skowron Volponi, M. A. (2018). trajr: An R package for characterisation of animal
        trajectories. Ethology, 124(6), 440-448. https://doi.org/10.1111/eth.12739.

    """
    if seed is None:
        np.random.seed(0)
    if angular_error_dist is None:
        angular_error_dist = np.random.normal(loc=0., scale=angular_error_sd, size=n)
    if linear_error_dist is None:
        linear_error_dist = np.random.normal(loc=0., scale=linear_error_sd, size=n)
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

    df = traja.TrajaDataFrame(data={'x': x, 'y': y})
    if fps in (0, None):
        raise Exception("fps must be greater than 0")
    df.fps = fps
    time = df.index / fps
    df['time'] = time
    df.spatial_units = spatial_units
    for key, value in kwargs.items():
        df.__dict__[key] = value
    # Update metavars
    metavars = dict(angular_error_sd=angular_error_sd, linear_error_sd=linear_error_sd)
    df.__dict__.update(metavars)

    return df


def _resample_time(trj, step_time):
    if not is_datetime_or_timedelta_dtype(trj.index):
        raise Exception(f"{trj.index.dtype} is not datetime or timedelta.")
    return trj.resample(step_time).agg({'x': np.mean, 'y': np.mean})


def resample_time(trj, step_time, new_fps=None):
    """Resample trajectory to consistent `step_time` intervals.

    Args:
        trj (:class:`~traja.TrajaDataFrame`): trajectory
        step_time (str): step time interval (eg, '1s')
        new_fps (bool, optional): new fps

    Results:
        trj (:class:`~traja.TrajaDataFrame`): trajectory


    .. doctest::

        >>> from traja import generate
        >>> from traja.utils import resample_time
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
    if time_col is 'index' and is_datetime64_any_dtype(trj.index):
        _trj = _resample_time(trj, step_time)
    elif time_col is 'index' and is_timedelta64_dtype(trj.index):
        _trj = _resample_time(trj, step_time)
    elif time_col:
        if isinstance(step_time, str):
            try:
                if '.' in step_time:
                    raise NotImplementedError("Fractional step time not implemented.")
            except Exception:
                raise NotImplementedError(f"Inferring from time format {step_time} not yet implemented.")
        _trj = trj.set_index(time_col)
        _trj.index = pd.to_timedelta(_trj.index, unit='s')
        _trj = _resample_time(_trj, step_time)
        _trj.reset_index(inplace=True)
    else:
        raise NotImplementedError(f"Time column ({time_col}) not of expected data type.")
    return _trj


def rotate(df, angle=0, origin=None):
    """Rotate a trajectory `angle` in radians.

    Args:
        trj: Traja.DataFrame
        angle

    Returns:
        trj: Traja.DataFrame

    .. note::

        Based on Lyle Scott's `implementation <https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302>`_.

    """
    trj = df.copy()
    # Calculate current orientation
    if isinstance(trj, traja.TrajaDataFrame):
        xy = df.traja.xy
    elif isinstance(trj, pd.DataFrame):
        trj = df[['x', 'y']]

    x, y = np.split(xy, [-1], axis=1)
    if origin is None:
        # Assume middle of x and y is origin
        origin = ((x.max() - x.min()) / 2, (y.max() - y.min()) / 2)

    offset_x, offset_y = origin
    new_coords = []

    for x, y in xy:
        adjusted_x = (x - offset_x)
        adjusted_y = (y - offset_y)
        cos_rad = math.cos(angle)
        sin_rad = math.sin(angle)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        new_coords.append((qx, qy))

    new_xy = np.array(new_coords)
    x, y = np.split(new_xy, [-1], axis=1)
    trj['x'] = x
    trj['y'] = y
    return trj


def from_df(df):
    """Convenience function for converting a :class:`pandas DataFrame<pandas.dataframe.DataFrame>` into a :class:`traja.main.TrajaDataFrame`.

    Args:
      df (:class:`pandas.DataFrame`): Trajectory as pandas `DataFrame`

    Returns:
      traj_df (:class:`~traja.main.TrajaDataFrame`): Trajectory

    .. doctest::

        >>> df = pd.DataFrame({'x':[0,1,2],'y':[1,2,3]})
        >>> traja.from_df(df)
           x  y
        0  0  1
        1  1  2
        2  2  3

    """
    traj_df = TrajaDataFrame(df)
    # Initialize metadata
    for var in traj_df._metadata:
        if not hasattr(traj_df, var):
            traj_df.__dict__[var] = None
    return traj_df


def from_xy(xy: np.ndarray):
    """Convenience function for initializing :class:`TrajaDataFrame<traja.main.TrajaDataFrame>` with x,y coordinates.

    Args:
        xy (:class:`numpy.ndarray`): x,y coordinates

    Returns:
        traj_df (:class:`~traja.main.TrajaDataFrame`): Trajectory as dataframe

    .. doctest::

        >>> import numpy as np
        >>> xy = np.array([[0,1],[1,2],[2,3]])
        >>> traja.from_xy(xy)
           x  y
        0  0  1
        1  1  2
        2  2  3

    """
    df = traja.TrajaDataFrame.from_records(xy, columns=['x', 'y'])
    return df


def fill_in_traj(trj):
    # FIXME: Implement
    return trj


def _get_time_col(trj):
    # Check if saved in metadata
    time_col = trj.__dict__.get('time_col', None)
    if time_col:
        return time_col
    # Check if index is datetime
    if is_datetime_or_timedelta_dtype(trj.index):
        return 'index'
    # Check if any column contains 'time'
    time_cols = [col for col in trj if 'time' in col.lower()]
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
