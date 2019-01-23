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
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


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

        # Downcast to float16
        float_cols = [c for c in df_test if 'float' in df_test[c].dtype]
        float16_cols = {c: np.float16 for c in float_cols}

        # Convert string columns to categories
        string_cols = [c for c in df_test if df_test[c].dtype == str]
        category_cols = {c: 'category' for c in string_cols}
        dtype = {**float16_cols, **category_cols}

        df = pd.read_csv(path,
                         infer_datetime_format=True,
                         date_parser=date_parser,
                         converters=converters,
                         dtype=dtype,
                         index_col=index_col,
                         )
        return df

    @property
    def xlim(self):
        return self._xlim

    @xlim.setter
    def xlim(self, xlim: tuple):
        self._xlim = xlim

    @property
    def ylim(self):
        return self._ylim

    @ylim.setter
    def ylim(self, ylim):
        self._ylim = ylim

    @property
    def xlabel(self):
        return self._xlabel

    @property
    def ylabel(self):
        return self._ylabel

    @xlabel.setter
    def xlabel(self, xlabel):
        self._xlabel = xlabel

    @ylabel.setter
    def ylabel(self, ylabel):
        self._ylabel = ylabel

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    def set(self, **kwargs):
        for key, value in kwargs.items():
            try:
                self.__setattr__(key, value)
            except Exception as e:
                logging.ERROR(f"Cannot set {key} to {value}")

    def plot(self, n_steps: int = 1000, days: tuple = None, **kwargs):
        """Plot trajectory for single animal over period.
            n_steps: int
            days: tuple of strings ('2018-01-01', '2019-02-01') or tuple of event-related ints (-1, 7)
            """
        start, end = None, None
        cbar_ticklabels = None
        __trj = self._trj[['x', 'y']]
        if days is not None:
            start, end = days
            if isinstance(start, str) and isinstance(end, str):
                # Datetime format
                mask = __trj.between(start, end, inclusive=True)
                verts = __trj.loc[mask].values
                cbar_ticklabels = (start, end)
            elif isinstance(start, int) and isinstance(end, int):
                # Range of days w.r.t. event, eg, for surgery, (-1, 7)
                # TODO: Implement this with reference to day of event (eg, `Days_from_surgery` column)
                raise NotImplementedError("Reference day will be column in `self._trj` or somewhere else")
        else:
            # Plot first `n_steps`
            start, end = 0, n_steps
            verts = __trj.iloc[:n_steps].values
            # cbar_ticklabels = (str(__trj.index[0]), str(__trj.index[n_steps]))
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        path = Path(verts, codes)

        fig, ax = plt.subplots()

        patch = patches.PathPatch(path, edgecolor='black', facecolor='none', lw=1)

        xs, ys = zip(*verts)

        n_steps = len(verts)
        colors = plt.cm.Greens_r(np.linspace(0, 1, n_steps))
        for i in range(len(xs)):
            ax.plot(xs[i], ys[i], 'x-', lw=1, color=colors[i], ms=2)

        ax.set_xlim(self._xlim)
        ax.set_ylim(self._ylim)
        ax.set_xlabel(self._xlabel)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.set_aspect('equal')

        # import ipdb;ipdb.set_trace()
        N = 21
        cmap = plt.get_cmap('Greens_r', N)
        norm = mpl.colors.Normalize(vmin=0, vmax=N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = plt.colorbar(sm)
        cbar_yticklabels = cbar.ax.get_yticklabels()
        if days is not None:
            cbar_yticklabels = [[start] + [''] * (len(cbar_yticklabels) - 2) + [end]]
        cbar_yticklabels = [__trj.index[i] for i in range(len(cbar_yticklabels))]
        cbar.ax.set_yticklabels(cbar_yticklabels)
        # cbar = plt.colorbar(ax=ax, ticks=[start, end])
        plt.tight_layout()
        plt.show()
        return ax

    def calc_distance(self):
        self._trj['distance'] = np.sqrt(np.power(self._trj.x.shift() - self._trj.x, 2) +
                                        np.power(self._trj.y.shift() - self._trj.y, 2))
        self._trj['dx'] = self._trj['x'].diff()
        self._trj['dy'] = self._trj['y'].diff()

    def calc_angle(self):
        if not {self._trj.columns}.issuperset({'dx', 'distance'}):
            self.calc_distance()
        self._trj['angle'] = np.rad2deg(np.arccos(np.abs(self._trj['dx']) / self._trj['distance']))

    def calc_heading(self):
        if not set(self._trj.columns).issuperset({'dx', 'dy'}):
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
    trj = track

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
    trj['displacementTime'] = trj.time - trj.time.iloc[0]

    ...


def traj(filepath, xlim=None, ylim=None, **kwargs):
    df_test = pd.read_csv(filepath, nrows=100)
    # Select first col with 'time_stamp' in name as index
    time_stamp_cols = [x for x in df_test.columns if 'time_stamp' in x]
    index_col = kwargs.pop('index_col', time_stamp_cols[0])

    df = pd.read_csv(filepath,
                     date_parser=kwargs.pop('data_parser',
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


def from_file(filepath, **kwargs):
    trj = pd.read_csv(filepath,
                       date_parser=kwargs.pop('data_parser',
                                              lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f')),
                       infer_datetime_format=kwargs.pop('infer_datetime_format', True),
                       parse_dates=kwargs.pop('parse_dates', True),
                       **kwargs)
    return trj




class Debug():
    def __init__(self, n_steps=1000):
        import glob
        from traja.main import TrajaAccessor, traj
        files = glob.glob('/Users/justinshenk/neurodata/data/raw_centroids_rev2/*')
        self.df = traj(files[10])
        self.df.traja.set(xlim=(-0.06, 0.06),
                          ylim=(-0.13, 0.13),
                          xlabel=("x (m)"),
                          ylabel=("y (m)"),
                          title="Cage trajectory")
        self.df.traja.plot(n_steps=n_steps)


def main(args):
    experiment = DVCExperiment(experiment_name='Stroke_olive_oil',
                               centroids_dir='/Users/justinshenk/neurodata/data/Stroke_olive_oil/dvc_tracking_position_raw/')
    experiment.aggregate_files()
    activity_files = experiment.get_activity_files()


def parse_arguments(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Load and analyze activity data')
    # TODO: Add cage dimensions argument
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_arguments(sys.argv)
    main()
