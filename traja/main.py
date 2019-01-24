#! /usr/local/env python3
import argparse
import glob
import logging
import multiprocessing as mp
import os
import psutil
import sys
import traja

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import unravel_index
from shapely.geometry import shape

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class TrajaDataFrame(pd.DataFrame):
    """A TrajaDataFrame object is a subclass of Pandas DataFrame.

    """

    _metadata = ['xlim', 'ylim', 'spatial_units', 'xlabel', 'ylabel', 'title', 'fps', 'time_units']

    def __init__(self, *args, **kwargs):
        super(TrajaDataFrame, self).__init__(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], TrajaDataFrame):
            args[0]._copy_attrs(self)

    @property
    def _constructor(self):
        return TrajaDataFrame

    def _copy_attrs(self, df):
        for attr in self._metadata:
            df.__dict__[attr] = getattr(self, attr, None)

    def __finalize__(self, other, method=None, **kwargs):
        """propagate metadata from other to self """
        # merge operation: using metadata of the left object
        if method == 'merge':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == 'concat':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        return self

    def copy(self, deep=True):
        """
        Make a copy of this TrajaDataFrame object
        Parameters
        ----------
        deep : boolean, default True
            Make a deep copy, i.e. also copy data
        Returns
        -------
        copy : TrajaDataFrame
        """
        data = self._data
        if deep:
            data = data.copy()
        return TrajaDataFrame(data).__finalize__(self)


@pd.api.extensions.register_dataframe_accessor("traja")
class TrajaAccessor(object):
    """Accessor for pandas DataFrame with trajectory-specific numerical and analytical functions."""

    def __init__(self, pandas_obj):
        # self.trajectory = self.read_csv(path, **kwargs)
        # return self.trajectory
        self._trj = pandas_obj

    def _strip(self, text):
        try:
            return text.strip()
        except AttributeError:
            return pd.to_numeric(text, errors='coerce')

    def set(self, **kwargs):
        for key, value in kwargs:
            try:
                self._trj.__dict__[key] = value
            except Exception as e:
                raise Exception(f"Exception {e} assigning df.{key} to {value}")

    @property
    def night(self, begin='19:00', end='7:00'):
        return self._trj.between_time(begin, end)

    @property
    def day(self, begin='7:00', end='19:00'):
        return self._trj.between_time(begin, end)

    def read_csv(self, path, **kwargs):
        index_col = kwargs.pop('index_col', None)

        date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f')

        df_test = pd.read_csv(path, nrows=100)

        if index_col not in df_test:
            logging.info(f'{index_col} not in {df_test.columns}')

        # Strip whitespace
        whitespace_cols = [c for c in df_test if ' ' in df_test[c].name]
        stripped_cols = {c: self._strip for c in whitespace_cols}
        # TODO: Add converters for processed 'datetime', 'x', etc. features
        converters = stripped_cols

        # Downcast to float32
        float_cols = [c for c in df_test if 'float' in df_test[c].dtype]
        float32_cols = {c: np.float32 for c in float_cols}

        # Convert string columns to categories
        string_cols = [c for c in df_test if df_test[c].dtype == str]
        category_cols = {c: 'category' for c in string_cols}
        dtype = {**float32_cols, **category_cols}

        df = pd.read_csv(path,
                         infer_datetime_format=True,
                         date_parser=date_parser,
                         converters=converters,
                         dtype=dtype,
                         index_col=index_col,
                         )
        return df

    def set(self, **kwargs):
        for key, value in kwargs.items():
            try:
                self.__setattr__(key, value)
            except Exception as e:
                logging.ERROR(f"Cannot set {key} to {value}")

    def plot(self, n_coords: int = None, days: tuple = None, **kwargs):
        """Plot trajectory for single animal over period.
            n_coords: int
            days: tuple of strings ('2018-01-01', '2019-02-01') or tuple of event-related ints (-1, 7)
            """
        GRAY = '#999999'
        if n_coords is not None and days is not None:
            raise NotImplementedError("Days and n_coords cannot both be specified.")

        start, end = None, None
        coords = self._trj[['x', 'y']]

        if days is not None:
            start, end = days
            if isinstance(start, str) and isinstance(end, str):
                # Datetime format
                mask = coords.between(start, end, inclusive=True)
                verts = coords.loc[mask].values
            elif isinstance(start, int) and isinstance(end, int):
                # Range of days w.r.t. event, eg, for surgery, (-1, 7)
                # TODO: Implement this with reference to day of event (eg, `Days_from_surgery` column)
                raise NotImplementedError("Reference day will be column in `self._trj` or somewhere else")
        elif n_coords is not None:
            # Plot first `n_coords`
            start, end = 0, n_coords
            verts = coords.iloc[:n_coords].values
        else:
            start, end = 0, len(coords)
            verts = coords.iloc[:end].values

        n_coords = len(verts)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        path = Path(verts, codes)

        fig, ax = plt.subplots()
        import ipdb;ipdb.set_trace()
        patch = patches.PathPatch(path, edgecolor=GRAY, facecolor='none', lw=3, alpha=0.3)
        ax.add_patch(patch)

        xs, ys = zip(*verts)

        colors = plt.cm.viridis(np.linspace(0, 1, n_coords))
        # for i in range(len(xs)):
        ax.scatter(xs, ys, c=colors, s=8, zorder=2, alpha=0.3)

        if coords.xlim is not None:
            ax.set_xlim(coords.xlim)
        else:
            ax.set_xlim((coords.x.min(), coords.x.max()))
        if coords.ylim is not None:
            ax.set_ylim(coords.ylim)
        else:
            ax.set_ylim((coords.y.min(), coords.y.max()))

        if kwargs.pop('invert_yaxis', None):
            plt.gca().invert_yaxis()
        ax.set_xlabel(coords.xlabel)
        ax.set_ylabel(coords.ylabel)
        ax.set_title(coords.title)
        ax.set_aspect('equal')

        N = 21  # bins
        cmap = plt.get_cmap('viridis', N)
        norm = mpl.colors.Normalize(vmin=0, vmax=N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = plt.colorbar(sm)
        cbar_yticklabels = cbar.ax.get_yticklabels()
        interval = n_coords // len(cbar_yticklabels)
        cbar_yticklabels = [coords.index[interval * i] for i in range(len(cbar_yticklabels))]
        cbar.ax.set_yticklabels(cbar_yticklabels)
        plt.tight_layout()
        plt.show()
        return ax

    def trip_grid(self, bins=16, log=False):
        """Generate a grid of time spent by point-to-cell gridding."""
        # TODO: Add kde-based method for line-to-cell gridding
        df = self._trj[['x', 'y']].dropna()
        x0, x1 = df.xlim or (df.x.min(), df.x.max())
        y0, y1 = df.ylim or (df.y.min(), df.y.max())
        aspect = (y1 - y0) / (x1 - x0)
        x_edges = np.linspace(x0, x1, num=bins)
        y_edges = np.linspace(y0, y1, num=int(bins / aspect))

        x, y = zip(*df.values)
        # # TODO: Remove redundant histogram calculation
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))
        fig, ax = plt.subplots()
        if log:
            hist = np.log(hist+np.e)
        image = plt.imshow(hist, interpolation='bilinear')
        # TODO: Set xticks and yticks to original data coordinates
        # TODO: Adjust colorbar ytick_labels to correspond with time
        cbar = plt.colorbar(image, ax=ax)
        plt.title("Time spent{}".format(' (Logarithmic)' if log else ''))
        plt.tight_layout()
        plt.show()
        # TODO: Add most common locations in grid
        # peak_index = unravel_index(hist.argmax(), hist.shape)

    def to_shapely(self):
        """Return shapely object for area, bounds, etc. functions."""
        df = self._trj[['x', 'y']].dropna()
        coords = df.values
        tracks_data = {'type': 'LineString',
                       'coordinates': coords}
        tracks_shape = shape(tracks_data)
        return tracks_shape

    def calc_distance(self):
        self._trj['distance'] = np.sqrt(np.power(self._trj.x.shift() - self._trj.x, 2) +
                                        np.power(self._trj.y.shift() - self._trj.y, 2))
        self._trj['dx'] = self._trj.x.diff()
        self._trj['dy'] = self._trj.y.diff()

    def calc_angle(self):
        if not set(self._trj.columns.tolist()).issuperset({'dx', 'distance'}):
            self.calc_distance()
        self._trj['angle'] = np.rad2deg(np.arccos(np.abs(self._trj['dx']) / self._trj['distance']))

    def rediscretize_points(self, R):  # WIP #
        """Resample a trajectory to a constant step length. R is rediscretized step length.
        Returns result, series of step coordinates.

        Based on the appendix in Bovet and Benhamou, (1988) and @JimMcL's trajr implementation.
        """
        # TODO: Test this method
        points = self._trj[['x', 'y']].dropna().values.astype('float64')
        n_points = len(points)
        # TODO: Implement with complex numbers
        result = np.empty((128, 2))
        p0 = points[0]
        result[0] = p0
        I = 0
        j = 1

        while j <= n_points:
            # Find the first point k for which |p[k] - p_0| >= R
            k = np.NaN
            for i in range(j, n_points):
                d = np.linalg.norm(points[i] - result[I])
                if d >= R:
                    k = i
                    break
            if np.isnan(k):
                # End of path
                break

            j = k

            # The next point lies on the segment p[k-1], p[k]
            XI = result[I][0]
            xk_1 = points[k - 1, 0]
            YI = result[I][1]
            yk_1 = points[k - 1, 1]

            a = 0 if XI - xk_1 > 0 else 1
            lambda_ = np.arctan((points[k, 1] - yk_1) / (points[k, 0] - xk_1)) + a * np.pi  # angle
            cos_l = np.cos(lambda_)
            sin_l = np.sin(lambda_)
            U = (XI - xk_1) * cos_l + (YI - yk_1) * sin_l
            V = (YI - yk_1) * cos_l + (XI - xk_1) * sin_l

            # Compute distance H between (X_{i+1}, Y_{i+1}) and (x_{k-1}, y_{k-1})
            H = U + np.sqrt(abs(R ** 2 - V ** 2))
            XIp1 = H * cos_l + xk_1
            YIp1 = H * sin_l + yk_1

            # Increase array size progressively to make the code run (significantly) faster
            if len(result) <= I + 1:
                result = np.concatenate((result, np.empty_like(result)))

            # Save the point
            result[I + 1] = np.array([XIp1, YIp1])
            I += 1

        # Truncate result
        result = result[:I + 2]
        return result

    def calc_heading(self):
        if not set(self._trj.columns.tolist()).issuperset({'dx', 'dy'}):
            self.calc_distance()
        # Get heading from angle
        mask = (self._trj['dx'] > 0) & (self._trj['dy'] >= 0)
        self._trj.loc[mask, 'heading'] = self._trj['angle'][mask]
        mask = (self._trj['dx'] >= 0) & (self._trj['dy'] < 0)
        self._trj.loc[mask, 'heading'] = -self._trj['angle'][mask]
        mask = (self._trj['dx'] < 0) & (self._trj['dy'] <= 0)
        self._trj.loc[mask, 'heading'] = -(180 - self._trj['angle'][mask])
        mask = (self._trj['dx'] <= 0) & (self._trj['dy'] > 0)
        self._trj.loc[mask, 'heading'] = (180 - self._trj['angle'])[mask]

    def calc_turn_angle(self):
        if 'heading' not in self._trj:
            self.calc_heading()
        self._trj['turn_angle'] = self._trj['heading'].diff()
        # Correction for 360-degree angle range
        self._trj.loc[self._trj.turn_angle >= 180, 'turn_angle'] -= 360
        self._trj.loc[self._trj.turn_angle < -180, 'turn_angle'] += 360


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


def traj(filepath, xlim=None, ylim=None, **kwargs):
    df_test = pd.read_csv(filepath, nrows=100)
    # Select first col with 'time_stamp' in name as index
    time_stamp_cols = [x for x in df_test.columns if 'time_stamp' in x]
    index_col = kwargs.pop('index_col', time_stamp_cols[0])

    df = pd.read_csv(filepath,
                     date_parser=kwargs.pop('date_parser',
                                            lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f')),
                     infer_datetime_format=kwargs.pop('infer_datetime_format', True),
                     parse_dates=kwargs.pop('parse_dates', True),
                     index_col=index_col,
                     **kwargs)
    if xlim is not None and isinstance(xlim, tuple):
        df.traja.xlim = xlim
    if ylim is not None and isinstance(ylim, tuple):
        df.traja.ylim = ylim
    return df


def read_file(filepath, **kwargs):
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)
    title = kwargs.pop('title', "Trajectory")
    spatial_units = kwargs.pop('spatial_units', 'm')
    xlabel = kwargs.pop('xlabel', f"x ({spatial_units})")
    ylabel = kwargs.pop('ylabel', f"y ({spatial_units})")
    fps = kwargs.pop('fps', None)
    index_col = kwargs.pop('index_col', None)
    if index_col is None:
        # Set index to first column containing 'time'
        df_test = pd.read_csv(filepath, nrows=10)
        time_cols = [col for col in df_test.columns if 'time' in col.lower()]
        if time_cols:
            index_col = time_cols[0] # Get first column
    if 'csv' in filepath:
        trj = pd.read_csv(filepath,
                          date_parser=kwargs.pop('date_parser',
                                                 lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f')),
                          infer_datetime_format=kwargs.pop('infer_datetime_format', True),
                          parse_dates=kwargs.pop('parse_dates', True),
                          index_col=index_col,
                          **kwargs)
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


class Debug():
    """Debug only.
    """

    def __init__(self, n_coords=1000):
        import glob
        from traja.main import TrajaAccessor, traj
        files = glob.glob('/Users/justinshenk/neurodata/data/raw_centroids_rev2/*')
        df = traja.read_file(files[10])
        df.traja.set(xlim=(-0.06, 0.06),
                     ylim=(-0.13, 0.13),
                     xlabel=("x (m)"),
                     ylabel=("y (m)"),
                     title="Cage trajectory")
        # df.traja.plot(n_coords=n_coords)
        result = df.traja.rediscretize_points(R=0.0002)


def main(args):
    experiment = traja.contrib.DVCExperiment(experiment_name='Stroke_olive_oil',
                                             centroids_dir='/Users/justinshenk/neurodata/data/Stroke_olive_oil/dvc_tracking_position_raw/')
    experiment.aggregate_files()
    activity_files = experiment.get_activity_files()


def parse_arguments(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Load and analyze trajectory data')
    # TODO: Add command line options
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv)
    main()
