#! /usr/local/env python3
import argparse
import glob
import logging
import multiprocessing as mp
import os
import psutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def totrajectory(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return Trajectory(result)
    return wrapper

class Trajectory():
    """Surrogate class for pandas DataFrame with trajectory-specific numerical and analytical functions."""
    def __init__(self, path, **kwargs):
        self.trajectory = self.read_csv(path, **kwargs)
        return self.trajectory

    @property
    def _constructor(self):
        return Trajectory

    def __repr__(self):
        return repr(self.contained)

    def __getitem__(self, item):
        result = self.contained[item]
        if isinstance(result, type(self.contained)):
            result = Trajectory(result)
        return result

    def __getattr__(self, item):
        result = getattr(self.contained, item)
        if callable(result):
            result = totrajectory(result)
        return result

    def _strip(self, text):
        try:
            return text.strip()
        except AttributeError:
            return pd.to_numeric(text, errors='coerce')

    def night(self):
        return self.trajectory.between_time('19:00','7:00')

    def read_csv(self, path, **kwargs):
        index_col = kwargs.pop('index_col', None)

        date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f')

        df_test = pd.read_csv(path, nrows=100)
        if index_col not in df_test:
            logging.info(f'{index_col} not in {df_test.columns}')

        whitespace_cols = [c for c in df_test if ' ' in df_test[c].name]
        stripped_cols = {c: self._strip for c in whitespace_cols}
        # TODO: Add converters for processed 'datetime', 'x', etc. features
        converters = stripped_cols

        float_cols = [c for c in df_test if df_test[c].dtype == 'float64']
        float16_cols = {c: np.float16 for c in float_cols}

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
        self.trajectory = df

    def from_csv(self, csvpath, **kwargs):
        df_test = pd.read_csv(csvpath, **kwargs, nrows=100)
        columns = [x.lower() for x in df_test.columns]
        assert set(columns).issuperset(set['x','y']), "Header does not contain 'x' and 'y'"
        df = pd.read_csv(csvpath, infer_datetime=True, **kwargs)
        self.trajectory = df

    def from_df(self, df):
        self.trajectory = df

    def plot(self, **kwargs):
        plt.plot(self.trajectory, **kwargs)

class DVCExperiment(object):
    def __init__(self, experiment_name, centroids_dir,
                 meta_filepath='/Users/justinshenk/neurodata/data/Stroke_olive_oil/DVC cageids HT Maximilian Wiesmann updated.xlsx',
                 cage_xmax = 0.058*2, cage_ymax= 0.125*2):
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
            ['position', 'Diet', 'Sham_or_Stroke', 'Stroke']
        ]
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

    def read_csv(self, path, index_col='time_stamp'):
        pass

    def get_cages(self):
        return [x for x in self.mouse_lookup.keys()]

    def get_ratios(self, file, angle_thresh, distance_thresh):
        ratios = []
        cage = file.split('/')[-1].split('_')[0]
        # Get x,y coordinates from centroids
        date_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S:%f')
        df = Trajectory(file, index_col='time_stamps_vec')[['x', 'y']]
        df.x = df.x.round(7)
        df.y = df.y.round(7)
        # Calculate euclidean distance (m) travelled
        df['distance'] = np.sqrt(np.power(df['x'].shift() - df['x'], 2) +
                                 np.power(df['y'].shift() - df['y'], 2))
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        # TODO: Replace with generic intervention method name and lookup logic
        surgery_date = self.get_stroke(cage)
        df['Days_from_surgery'] = (df.index - surgery_date).days

        # Calculate angle w.r.t. x axis
        df['angle'] = np.rad2deg(np.arccos(np.abs(df['dx']) / df['distance']))
        # Get heading from angle
        mask = (df['dx'] > 0) & (df['dy'] >= 0)
        df.loc[mask, 'heading'] = df['angle'][mask]
        mask = (df['dx'] >= 0) & (df['dy'] < 0)
        df.loc[mask, 'heading'] = -df['angle'][mask]
        mask = (df['dx'] < 0) & (df['dy'] <= 0)
        df.loc[mask, 'heading'] = -(180 - df['angle'][mask])
        mask = (df['dx'] <= 0) & (df['dy'] > 0)
        df.loc[mask, 'heading'] = (180 - df['angle'])[mask]
        df['turn_angle'] = df['heading'].diff()
        # Correction for 360-degree angle range
        df.loc[df.turn_angle >= 180, 'turn_angle'] -= 360
        df.loc[df.turn_angle < -180, 'turn_angle'] += 360
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
                  (left + right) > 0]  # fix div by 0 errror
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

    def get_centroid(self, cage):
        path = os.path.join(self.outdir, 'centroids', cage)
        df = self.read_csv(path)
        return df

    def plot_position_heatmap(self, cage):
        from numpy import unravel_index
        # TODO: Generate from y in +-0.12, x in +-0.058
        x_edges = np.array([-0.1201506, -0.11524541, -0.11034022, -0.10543504, -0.10052985,
                            -0.09562466, -0.09071947, -0.08581429, -0.0809091, -0.07600391,
                            -0.07109872, -0.06619353, -0.06128835, -0.05638316, -0.05147797,
                            -0.04657278, -0.0416676, -0.03676241, -0.03185722, -0.02695203,
                            -0.02204684, -0.01714166, -0.01223647, -0.00733128, -0.00242609,
                            0.00247909, 0.00738428, 0.01228947, 0.01719466, 0.02209984,
                            0.02700503, 0.03191022, 0.03681541, 0.0417206, 0.04662578,
                            0.05153097, 0.05643616, 0.06134135, 0.06624653, 0.07115172,
                            0.07605691, 0.0809621, 0.08586729, 0.09077247, 0.09567766,
                            0.10058285, 0.10548804, 0.11039322, 0.11529841, 0.1202036])

        y_edges = np.array([-0.05804244, -0.05567644, -0.05331044, -0.05094444, -0.04857844,
                            -0.04621243, -0.04384643, -0.04148043, -0.03911443, -0.03674843,
                            -0.03438243, -0.03201643, -0.02965043, -0.02728443, -0.02491843,
                            -0.02255242, -0.02018642, -0.01782042, -0.01545442, -0.01308842,
                            -0.01072242, -0.00835642, -0.00599042, -0.00362442, -0.00125842,
                            0.00110759, 0.00347359, 0.00583959, 0.00820559, 0.01057159,
                            0.01293759, 0.01530359, 0.01766959, 0.02003559, 0.02240159,
                            0.0247676, 0.0271336, 0.0294996, 0.0318656, 0.0342316,
                            0.0365976, 0.0389636, 0.0413296, 0.0436956, 0.0460616,
                            0.04842761, 0.05079361, 0.05315961, 0.05552561, 0.05789161])

        df = self.get_centroid(cage)
        x, y = zip(*df[['x', 'y']].values)
        # TODO: Remove redundant histogram calculation
        H, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))
        cmax = H.flatten().argsort()[-2]  # Peak point is too hot, bug?

        fig, ax = plt.subplots()
        hist, x_edges, y_edges, image = ax.hist2d(np.array(y), np.array(x),
                                                  bins=[np.linspace(df.y.min(), df.y.max(), 50),
                                                        np.linspace(df.x.min(), df.x.max(), 50)],
                                                  cmax=cmax)
        ax.colorbar()
        # peak_index = unravel_index(hist.argmax(),hist.shape)

    def get_activity_files(self):
        activity_dir = os.path.join(self.basedir, 'data', self.experiment_name, 'dvc_activation', '*')
        activity_files = glob.glob(activity_dir)
        assert activity_files, "No activity files"
        return activity_files

    def aggregate_files(self):
        """Aggregate cage files into csvs"""
        os.makedirs(os.path.join(self.outdir,'centroids'), exist_ok=True)
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
        activity_csv = os.path.join(self.outdir,'daily_activity.csv')
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
