#! /usr/local/env python3
import argparse
import glob
import logging
import math
import os
import sys
from collections import OrderedDict

import traja
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import scipy

from traja import TrajaDataFrame
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from scipy.spatial.distance import directed_hausdorff, euclidean
from numpy import unravel_index
from shapely.geometry import shape

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


def stylize_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)


def shift_xtick_labels(xtick_labels, first_index=None):
    for idx, x in enumerate(xtick_labels):
        label = x.get_text()
        xtick_labels[idx].set_text(str(int(label) + 1))
        if first_index is not None:
            xtick_labels[0] = first_index
    return xtick_labels

def fill_in_traj(trj):
    # FIXME: Implement
    return trj

def smooth_sg(trj, w = None, p = 3):
    """Savitzky-Golay filtering.

    Args:
      trj: 
      w:  (Default value = None)
      p:  (Default value = 3)

    Returns:

    """
    if w is None:
        w = p + 3 - p % 2

    if (w % 2 != 1):
        raise Exception(f"Invalid smoothing parameter w ({w}): n must be odd")
    trj.x = scipy.signal.savgol_filter(trj.x, window_length = w, polyorder=p, axis=0)
    trj.y = scipy.signal.savgol_filter(trj.y, window_length = w, polyorder=p, axis=0)
    trj = fill_in_traj(trj)
    return trj

def angles(trj, lag = 1, compass_direction = None):
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
      trj: 

    Returns:

    """
    raise NotImplementedError()


def polar_to_z(r, theta):
    """Converts polar coordinates `z` and `theta` to complex number `z`.

    Args:
      r: 
      theta: 

    Returns:

    """
    return r * np.exp(1j * theta)


def cartesian_to_polar(xy):
    """Convert np.array `xy` to polar coordinates `r` and `theta`.

    Args:
      xy (np.ndarray): x,y coordinates

    Returns:
        r, theta (tuple of float): step-length and angle

    """
    assert xy.ndim == 2, f"Dimensions are {xy.ndim}, expecting 2"
    x, y = np.split(xy,[-1], axis=1)
    x, y = np.squeeze(x), np.squeeze(y)
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r, theta

def expected_sq_displacement(trj, n = None, eqn1= True, compass_direction = None):
    # TODO: Fix and test implementation
    sl = step_lengths(trj)
    ta = angles(trj, compass_direction = compass_direction)
    l = np.mean(sl)
    l2 = np.mean(sl ^ 2)
    c = np.mean(np.cos(ta))
    s = np.mean(np.sin(ta))
    s2 = s ^ 2

    if eqn1:
        # Eqn 1
        alpha = np.arctan2(s, c)
        gamma = ((1 - c)^2 - s2) * np.cos((n + 1) * alpha) - 2 * s * (1 - c) * np.sin((n + 1) * alpha)
        esd = n * l2 + 2 * l^2 * ((c - c^2 - s2) * n  - c) / ((1 - c)^2 + s2) + 2 * l^2 * ((2 * s2 + (c + s2) ^ ((n + 1) / 2)) / ((1 - c)^2 + s2)^2) * gamma
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


def distance(A, B, method='dtw'):
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
            pip install fastdtw.""")
        distance, path = fastdtw(A, B, dist=euclidean)
        return distance


def generate(n=1000, random=True, step_length=2,
             angular_error_sd=0.5,
             angular_error_dist=None,
             linear_error_sd=0.2,
             linear_error_dist=None,
             fps=50,
             spatial_units='m',
             **kwargs):
    """Generates a trajectory.

    If ``random``` is ``True``, the trajectory will
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
    the ``angularErrorDist`` and/or ``linearErrorDist`` parameters.
    
    The initial angle (for a random walk) or the intended direction (for a
    directed walk) is ``0`` radians. The starting position is ``(0, 0)``.
    
    .. note::

        Author: Jim McLean (trajr), ported to Python by Justin Shenk.

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

    """

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
        coords = np.array([complex(0), np.cumsum(steps)], dtype=np.complex)

    x = coords.real
    y = coords.imag

    df = pd.DataFrame(data={'x': x, 'y': y})
    df.fps = fps
    df.spatial_units = spatial_units
    for key, value in kwargs:
        df.__dict__[key] = value
    return df


def from_df(df):
    """Convenience function for converting a Pandas DataFrame into a TrajaDataFrame.

    Args:
      df: pandas DataFrame

    Returns:
      TrajaDataFrame
      
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
    """Convenience function for initializing TrajaDataFrame with x,y coordinates.

    Args:
      xy: np.ndarray: 

    Returns:

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


def read_file(filepath, **kwargs):
    """Convenience method wrapping pandas `read_csv` and initializing metadata.

    Args:
      filepath: 
      **kwargs: 

    Returns:

    """

    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)
    title = kwargs.pop('title', "Trajectory")
    spatial_units = kwargs.pop('spatial_units', 'm')
    xlabel = kwargs.pop('xlabel', f"x ({spatial_units})")
    ylabel = kwargs.pop('ylabel', f"y ({spatial_units})")
    fps = kwargs.pop('fps', None)
    date_parser = kwargs.pop('data_parser', None)

    # TODO: Set index to first column containing 'time'
    df_test = pd.read_csv(filepath, nrows=10, parse_dates=True, infer_datetime_format=True)

    # Strip whitespace
    whitespace_cols = [c for c in df_test if ' ' in df_test[c].name]
    stripped_cols = {c: lambda x:x.strip() for c in whitespace_cols}
    converters = {**stripped_cols, **kwargs.pop('converters',{})}

    # Downcast to float32 # TODO: Benchmark float32 vs float64 for very big datasets
    float_cols = [c for c in df_test if 'float' in df_test[c].dtype]
    float32_cols = {c: np.float32 for c in float_cols}

    # Convert string columns to categories
    string_cols = [c for c in df_test if df_test[c].dtype == str]
    category_cols = {c: 'category' for c in string_cols}
    dtype = {**float32_cols, **category_cols, **kwargs.pop('dtype', {})}

    time_cols = [col for col in df_test.columns if 'time' in col.lower()]

    if 'csv' in filepath:
        trj = pd.read_csv(filepath,
                          date_parser=date_parser,
                          infer_datetime_format=kwargs.pop('infer_datetime_format', True),
                          parse_dates=kwargs.pop('parse_dates', True),
                          converters=converters,
                          dtype=dtype,
                          **kwargs)
        if time_cols:
            time_col = time_cols[0]
            trj.rename(columns={time_col: 'time'})
        else:
            time = (trj.index) / fps
            trj['time'] = time
    else:
        # TODO: Implement for HDF5 and .npy files.
        raise NotImplementedError("Non-csv's not yet implemented")

    trj = TrajaDataFrame(trj)
    # Set meta properties of TrajaDataFrame
    trj.xlim = xlim
    trj.ylim = ylim
    trj.spatial_units = spatial_units
    trj.title = title
    trj.xlabel = xlabel
    trj.ylabel = ylabel
    trj.fps = fps
    return trj
