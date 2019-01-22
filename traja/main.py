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


class DVCExperiment(object):
    def __init__(self, experiment_name, centroids_dir,
                 meta_filepath='/Users/justinshenk/neurodata/data/Stroke_olive_oil/DVC cageids HT Maximilian Wiesmann updated.xlsx',
                 cage_xmax=0.058 * 2, cage_ymax=0.125 * 2):
        # TODO: Fix in prod version
        self._init()
        self.basedir = '/Users/justinshenk/neurodata/'
        self._cpu_count = psutil.cpu_count()
        self.centroids_dir = centroids_dir
        search_path = glob.glob(os.path.join(centroids_dir, '*'))
        self.centroids_files = sorted(
            [x.split('/')[-1] for x in search_path if 'csv' in x and 'filelist' not in x])
        self.mouse_lookup = self.load_meta(meta_filepath)
        self.cage_xmax = cage_xmax
        self.cage_ymax = cage_ymax
        self.experiment_name = experiment_name
        self.outdir = os.path.join(self.basedir, 'output', self._str2filename(experiment_name))
        self.cages = self.get_cages(centroids_dir)

    def _init(self):
        plt.rc('font', family='serif')

    @staticmethod
    def _str2filename(string):
        filename = string.replace(' ', '_')
        # TODO: Implement filename security
        filename = filename.replace('/', '')
        return filename

    def get_weekly_activity(self):
        activity = self.get_daily_activity()
        weekly_list = []

        for week in range(-3, 5):
            for group in activity['Group+Diet'].unique():
                for period in ['Daytime', 'Nighttime']:
                    df = activity[(activity.Days_from_surgery >= week * 7 + 1)  # ...-6, 1, 8, 15...
                                  & (activity.Days_from_surgery < (week + 1) * 7 + 1)  # ...1, 8, 15, 21...
                                  & (activity['Group+Diet'] == group)
                                  & (activity.Period == period)].groupby(['Cage']).Activity.mean().to_frame()
                    df['Group+Diet'] = group
                    df['Week'] = week
                    df['Period'] = period
                    # df['Cohort'] = [get_cohort(x) for x in df.index]
                    weekly_list.append(df)
        weekly = pd.concat(weekly_list)
        return weekly

    def plot_weekly(self, weekly, groups):
        for group in groups:
            fig, ax = plt.subplots(figsize=(4, 3))
            for period in ['Daytime', 'Nighttime']:
                sns.pointplot(x='Week', y='Activity', hue='Cohort',
                              data=weekly[(weekly['Group+Diet'] == group) & (weekly['Period'] == period)].groupby(
                                  'Activity').mean().reset_index(),
                              ci=68)
            plt.title(group)
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels[:2], handles[:2]), key=lambda t: t[0]))
            ax.legend(handles, labels)
            plt.tight_layout()
        plt.show()

    def get_presurgery_average_weekly_activity(self):
        """Average pre-stroke weeks into one point."""
        pre_average_weekly_act = os.path.join(self.outdir, 'pre_average_weekly_act.csv')
        if not os.path.exists(pre_average_weekly_act):
            weekly = self.get_weekly_activity()
            for period in ['Daytime', 'Nighttime']:
                for cage in self.get_cages():
                    mean = weekly[
                        (weekly.index == cage) & (weekly.Week < 0) & (weekly.Period == period)].Activity.mean()
                    weekly.loc[
                        (weekly.index == cage) & (weekly.Week < 0) & (weekly.Period == period), 'Activity'] = mean
        else:
            weekly = self.read_csv(pre_average_weekly_act)
            return weekly

    def norm_weekly_activity(self, weekly):
        # Normalize activity
        weekly['Normed_Activity'] = 0
        for period in ['Daytime', 'Nighttime']:
            for cage in self.get_cages():
                df_night = weekly[(weekly['Week'] >= -1) & (weekly.index == cage) & (weekly.Period == 'Nighttime')]
                df = weekly[(weekly['Week'] >= -1) & (weekly.index == cage) & (weekly.Period == period)]
                assert df.Week.is_monotonic_increasing == True, "Not monotonic"
                normed = [x / df_night.Activity.values[0] for x in df.Activity.values]
                weekly.loc[(weekly.index == cage) & (weekly.Period == period) & (
                        weekly.Week >= -1), 'Normed_Activity'] = normed
        return weekly

    def _stylize_axes(self, ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.xaxis.set_tick_params(top='off', direction='out', width=1)
        ax.yaxis.set_tick_params(right='off', direction='out', width=1)

    def _shift_xtick_labels(self, xtick_labels, first_index=None):
        for idx, x in enumerate(xtick_labels):
            label = x.get_text()
            xtick_labels[idx].set_text(str(int(label) + 1))
            if first_index is not None:
                xtick_labels[0] = first_index
        return xtick_labels

    def _norm_daily_activity(self, activity):
        norm_daily_activity_csv = os.path.join(self.outdir, 'norm_daily_activity.csv')
        if not os.path.exists(norm_daily_activity_csv):
            activity['Normed_Activity'] = 0
            for period in ['Daytime', 'Nighttime']:
                for cage in self.get_cages():
                    # Get prestroke
                    prestroke_night_average = activity[(activity.Days_from_surgery <= -1) & (activity.Cage == cage) & (
                            activity.Period == 'Nighttime')].Activity.mean()
                    df = activity[
                        (activity.Days_from_surgery >= -1) & (activity.Cage == cage) & (activity.Period == period)]
                    assert df.Days_from_surgery.is_monotonic_increasing == True, "Not monotonic"
                    mean = activity[(activity.Days_from_surgery <= -1) & (activity.Cage == cage) & (
                            activity.Period == period)].Activity.mean()
                    df.loc[(df.Cage == cage) & (df.Period == period) & (df.Days_from_surgery == -1), 'Activity'] = mean
                    normed = [x / prestroke_night_average for x in df.Activity.values]
                    activity.loc[(activity.Cage == cage) & (activity.Period == period) & (
                            activity.Days_from_surgery >= -1), 'Normed_Activity'] = normed
            activity.to_csv(norm_daily_activity_csv)
        else:
            activity = pd.read_csv(norm_daily_activity_csv)
        return activity

    def plot_daily_normed_activity(self):
        activity = self.get_daily_activity()
        activity = self._norm_daily_activity(activity)

    def plot_weekly_normed_activity(self, presurgery_average=True):
        """Plot weekly normed activity. Optionally, average presurgery points."""
        if presurgery_average:
            weekly = self.get_presurgery_average_weekly_activity()
            # for cohort in [2,4]:
            fig, ax = plt.subplots(figsize=(6.25, 3.8))
            hue_order = weekly['Group+Diet'].unique()
            group_cnt = len(hue_order)
            for period in ['Daytime', 'Nighttime']:
                linestyles = ['--'] * group_cnt if period is 'Daytime' else ['-'] * group_cnt
                sns.pointplot(x='Week', y='Normed_Activity', hue='Group+Diet', data=weekly[(weekly.Week >= -1) &
                                                                                           (weekly.Period == period)],
                              #                                                                               (weekly.Cohort==cohort)],
                              palette=['k', 'gray', 'C0', 'C1'][:group_cnt],
                              linestyles=linestyles,
                              # hue_order=['Sham - Control', 'Sham - HT', 'Stroke - Control', 'Stroke - HT'],
                              hue_order=hue_order,
                              markers=["d", "s", "^", "x"][:group_cnt],  # TODO: Generalize for larger sets
                              dodge=True,
                              ci=68)
            ax.set_xlabel('Weeks from Surgery')
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels[:4], handles[:4]), key=lambda t: t[0]))
            ax.legend(handles, labels)
            self._stylize_axes(ax)
            fig.set_facecolor('white')
            xtick_labels = ax.get_xticklabels()
            xtick_labels = self._shift_xtick_labels(xtick_labels, 'Pre-surgery')

            plt.ylabel('Normalized Activity')
            ax.set_xticklabels(xtick_labels)
            plt.title('Normalized Activity')
            plt.show()

    @staticmethod
    def load_meta(self, meta_filepath):
        # TODO: Generalize
        mouse_data = pd.read_excel(meta_filepath)[
            ['position', 'Diet', 'Sham_or_Stroke', 'Stroke']]
        mouse_data['position'] = mouse_data['position'].apply(lambda x: x[1] + x[0].zfill(2))
        return mouse_data.set_index('position').to_dict('index')

    @staticmethod
    def get_diet(self, cage):
        return self.mouse_lookup[cage]['Diet']

    @staticmethod
    def get_group(self, cage):
        return self.mouse_lookup[cage]['Sham_or_Stroke']

    @staticmethod
    def get_stroke(self, cage):
        return self.mouse_lookup[cage]['Stroke']

    @staticmethod
    def get_group_and_diet(self, cage):
        diet = self.get_diet(cage)
        surgery = self.get_group(cage)
        return f"{'Sham' if surgery is 1 else 'Stroke'} - {'Control' if diet is 1 else 'HT'}"

    @staticmethod
    def get_cohort(self, cage):
        # TODO: Generalize
        return self.mouse_lookup[cage]['Stroke'].month

    @staticmethod
    def get_cages(self, centroid_dir):
        # FIXME: Complete implementation
        return ['A04']

    # def read_csv(self, path, index_col='time_stamp'):
    #     pass

    def get_cages(self):
        return [x for x in self.mouse_lookup.keys()]

    def get_turn_ratios(self, file, angle_thresh, distance_thresh):
        ratios = []
        cage = file.split('/')[-1].split('_')[0]
        # Get x,y coordinates from centroids
        date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f')
        df = traja.from_file(file, index_col='time_stamps_vec')[['x', 'y']]
        # df.x = df.x.round(7)
        # df.y = df.y.round(7)
        df.traja.calc_distance()  # adds 'distance' column
        # TODO: Replace with generic intervention method name and lookup logic
        surgery_date = self.get_stroke(cage)
        df['Days_from_surgery'] = (df.index - surgery_date).days

        df.traja.calc_turn_angle()  # adds 'turn_angle' column
        #     df['turn_angle'].where((df['distance']>1e-3) & ((df.turn_angle > -15) & (df.turn_angle < 15))).hist(bins=30)
        #     df['turn_bias'] = df['turn_angle'] / .25 # 0.25s
        # Only look at distances over .01 meters, resample to minute intervals
        distance_mask = df['distance'] > (distance_thresh)
        angle_mask = ((df.turn_angle > angle_thresh) & (df.turn_angle < 90)) | (
                (df.turn_angle < -angle_thresh) & (df.turn_angle > -90))

        day_mask = (df.index.hour >= 7) & (df.index.hour < 19)
        day_mean = df.loc[distance_mask & angle_mask & day_mask, 'turn_angle'].dropna()
        night_mean = df.loc[distance_mask & angle_mask & ~day_mask, 'turn_angle'].dropna()
        right_turns_day = day_mean[day_mean > 0].shape[0]
        left_turns_day = day_mean[day_mean < 0].shape[0]
        right_turns_night = night_mean[night_mean > 0].shape[0]
        left_turns_night = night_mean[night_mean < 0].shape[0]
        ratios.append((df.Days_from_surgery[0], right_turns_day, left_turns_day, False))
        ratios.append((df.Days_from_surgery[0], right_turns_night, left_turns_night, True))

        ratios = [(day, right, left, period) for day, right, left, period in ratios if
                  (left + right) > 0]  # fix div by 0 error
        return ratios
        #     days = [day for day, _, _, nighttime in ratios if nighttime]

        #     laterality = [right_turns/(left_turns+right_turns) for day, right_turns, left_turns, nighttime in ratios if nighttime]
        #     fig, ax = plt.subplots()
        #     ax.plot(days, laterality, label='Laterality')
        #     ax.set(title=f"{cage} laterality index (right/right+left)\nDistance threshold: 0.25 cm\nAngle threshold: {thresh}\nRight turn is > 0.5\n{get_diet(cage)}",
        #           xlabel="Days from surgery",
        #           ylabel="Laterality index")
        #     ax.legend()
        #     ax.set_ylim((0,1.0))
        #     ax2 = ax.twinx()
        #     ax2.plot(days, [right+left for _, right, left, nighttime in ratios if nighttime],color='C1', label='Number of turns')
        #     ax2.set_ylabel('Number of turns')
        #     ax2.legend()
        #     plt.show()

    def calculate_turns(self, angle_thresh=30, distance_thresh=0.0025):
        ratio_dict = {}
        for cage in self.get_cages():
            ratio_dict[cage] = []

            with mp.Pool(processes=self._cpu_count) as p:
                args = [(file, angle_thresh, distance_thresh) for file in self.centroids_files if cage in file]
                ratios = p.starmap(self.get_ratios, args)
                ratio_dict[cage].append(ratios)
                logging.info(f'Processed {cage}')

        turn_ratio_csv = os.path.join(self.outdir,
                                      f'ratios_angle-{angle_thresh}_distance-{distance_thresh}_period_turnangle.npy')
        np.save(turn_ratio_csv, ratio_dict)
        logging.info(f'Saved to {turn_ratio_csv}')
        return ratio_dict

    def get_coords(self, cage):
        path = os.path.join(self.outdir, 'centroids', cage)
        df = traja.from_file(path)
        return df

    def plot_position_heatmap(self, cage, bins=20):
        from numpy import unravel_index
        # TODO: Generate from y in +-0.12, x in +-0.058
        try:
            x0, x1 = self._trj.traja.xlim
            y0, y1 = self._trj.traja.ylim
        except:
            raise NotImplementedError("Not yet implemented automated heatmap binning")
        x_edges = np.linspace(x0, x1, num=bins)
        y_edges = np.linspace(y0, y1, num=bins)

        trj = self.get_coords(cage)
        x, y = zip(*trj[['x', 'y']].values)
        # TODO: Remove redundant histogram calculation
        H, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))
        cmax = H.flatten().argsort()[-2]  # Peak point is too hot, bug?

        fig, ax = plt.subplots()
        hist, x_edges, y_edges, image = ax.hist2d(np.array(y), np.array(x),
                                                  bins=[np.linspace(trj.y.min(), trj.y.max(), 50),
                                                        np.linspace(trj.x.min(), trj.x.max(), 50)],
                                                  cmax=cmax)
        ax.colorbar()
        ax.set_aspect('equal')
        plt.show()
        # peak_index = unravel_index(hist.argmax(),hist.shape)

    def get_activity_files(self):
        activity_dir = os.path.join(self.basedir, 'data', self.experiment_name, 'dvc_activation', '*')
        activity_files = glob.glob(activity_dir)
        assert activity_files, "No activity files"
        return activity_files

    def aggregate_files(self):
        """Aggregate cage files into csvs"""
        os.makedirs(os.path.join(self.outdir, 'centroids'), exist_ok=True)
        for cage in self.centroid_files:
            logging.info(f'Processing {cage}')
            # Check for aggregated cage file (eg, 'A04.csv')
            cage_path = os.path.join(self.outdir, 'centroids', f'{cage}.csv')
            if os.path.exists(cage_path):
                continue
            # Otherwise, generate one
            search_path = os.path.join(self.centroids_dir, cage, '*.csv')
            files = glob.glob(search_path)

            days = []
            for file in files:
                _df = self.read_csv(file)
                _df.columns = [x.strip() for x in _df.columns]
                days.append(_df)
            df = pd.concat(days).sort_index()
            #     for col in ['x','y','distance']:
            #         df.applymap(lambda x: x.str.strip() if isinstance(x,str) else x)
            #         df[col] = pd.to_numeric(df[col],errors='coerce')
            cage_path = os.path.join(self.outdir, 'centroids', f'{cage}.csv')
            df.to_csv(cage_path)
            logging.info(f'saved to {cage_path}')
        # activity_df = self.read_csv('data/Stroke_olive_oil/dvc_activation/A04.csv', index_col='time_stamp_start')
        return

    def _get_ratio_dict(self, angle=30, distance=0.0025):
        npy_path = os.path.join(self.outdir, 'ratios_angle-{angle}_distance-{distance}_period_turnangle.npy')
        r = np.load(npy_path)
        ratio_dict = r.item(0)
        return ratio_dict

    def get_cage_laterality(self, cage):
        ratio_dict = self._get_ratio_dict()
        ratios = ratio_dict[cage]
        ratios = [x for x in ratios if (x[1] + x[2] > 0)]
        days = [day for day, _, _, nighttime in ratios if nighttime]

        laterality = [right_turns / (left_turns + right_turns) for day, right_turns, left_turns, nighttime in ratios
                      if nighttime]
        fig, ax = plt.subplots()
        ax.plot(days, laterality, label='Laterality')
        ax.set(
            title=f"{cage} laterality index (right/right+left)\nDistance threshold: 0.25 cm\nAngle threshold: {thresh}\nRight turn is > 0.5\n{self.get_diet(cage)}",
            xlabel="Days from surgery",
            ylabel="Laterality index")
        ax.legend()
        ax.set_ylim((0, 1.0))
        ax2 = ax.twinx()
        ax2.plot(days, [right + left for _, right, left, nighttime in ratios if nighttime], color='C1',
                 label='Number of turns')
        ax2.set_ylabel('Number of turns')
        ax2.legend()
        plt.show()

    def get_daily_activity(self):
        activity_csv = os.path.join(self.outdir, 'daily_activity.csv')
        if not os.path.exists(activity_csv):
            print(f"Path {activity_csv} does not exist, creating dataframe")
            activity_list = []
            col_list = [f'e{i:02}' for i in range(1, 12 + 1)]  # electrode columns
            # Iterate over minute activations
            search_path = os.path.join(self.basedir, 'data', self.experiment_name, 'dvc_activation', '*.csv')
            minute_activity_files = sorted(
                glob.glob(search_path))
            for cage in minute_activity_files:
                cage_id = os.path.split(cage)[-1].split('.')[0]
                # TODO: Fix in final
                assert len(cage_id) == 3, logging.error(f"{cage_id} length != 3")
                # Read csv
                cage_df = pd.read_csv(cage, index_col='time_stamp_start',
                                      date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f'))
                # Make csv with columns for cage+activity+day+diet+surgery
                cage_df['Activity'] = cage_df[col_list].sum(axis=1)
                day = cage_df.Activity.between_time('7:00', '19:00').resample('D').sum().to_frame()
                day['Cage'] = cage_id
                day['Period'] = 'Daytime'
                day['Surgery'] = self.get_stroke(cage_id)
                day['Diet'] = self.get_diet(cage_id)
                day['Group'] = self.get_group(cage_id)
                day['Days'] = [int(x) for x in range(len(day.index))]
                activity_list.append(day)

                night = cage_df.Activity.between_time('19:00', '7:00').resample('D').sum().to_frame()
                night['Cage'] = cage_id
                night['Period'] = 'Nighttime'
                night['Surgery'] = self.get_stroke(cage_id)
                night['Diet'] = self.get_diet(cage_id)
                night['Group'] = self.get_group(cage_id)
                night['Days'] = [int(x) for x in range(len(night.index))]
                activity_list.append(night)

            activity = pd.concat(activity_list)
            activity.to_csv(activity_csv)
        else:
            activity = pd.read_csv(activity_csv,
                                   index_col='time_stamp_start',
                                   parse_dates=['Surgery', 'time_stamp_start'],
                                   infer_datetime_format=True)
        return activity


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
