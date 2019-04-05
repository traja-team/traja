#! /usr/local/env python3
import argparse
import glob
import logging
import multiprocessing as mp
import os
import re

from scipy.stats import ttest_ind, ttest_rel
import statsmodels.api as sm
import statsmodels.formula.api as smf

import traja
from traja.plotting import *

import matplotlib.style as style
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    raise ImportError(
        "##########################\n"
        "seaborn is not installed. \n"
        "install it with\n"
        "  pip install seaborn\n"
        "\n"
        "##########################"
    )

DARK = "Dark"
LIGHT = "Light"


def glm(
    daily: pd.DataFrame,
    response_var: str = "displacement",
    cov_struct: Optional[str] = None,
):
    """Generalized linear model for daily response."""

    kwargs = {}

    if cov_struct == "auto":
        cov_struct = sm.cov_struct.Autoregressive()
        kwargs["cov_struct"] = cov_struct
    elif cov_struct == "exchangable":
        cov_struct = sm.cov_struct.Exchangeable()
        kwargs["cov_struct"] = cov_struct
    day_range = f"{daily.days_from_surgery.min()}-{daily.days_from_surgery.max()}"
    print(
        f"Fitting generalized linear model on daily data. Days {day_range}\n"
        f"Model: {response_var} ~ days_from_surgery * diet."
        + (f" Covariance structure ({cov_struct})" if cov_struct else "")
    )

    sig_pvalues = []

    for period in [DARK, LIGHT, "Overall"]:
        subset = daily[daily.period == period] if period != "Overall" else daily
        md = smf.gee(
            f" {response_var} ~ days_from_surgery * diet",
            data=subset,
            groups="cage",
            **kwargs,
        )
        mdf = md.fit()
        pvalues = mdf.pvalues
        for factor, pvalue in pvalues.items():
            if pvalue < 0.05 and factor != "Intercept":
                print(
                    f"{factor} effect on {response_var} ({period}) - p-value: {pvalue:.4f}"
                )
                sig_pvalues.append((factor, pvalue))
    if not sig_pvalues:
        print("No significant effect found.")
    return sig_pvalues


class DVCExperiment:
    """Mouse tracking data analysis.
    experiment_name
    centriods_dir
    meta_filepath
    cage_xmax
    cage_ymax

    Args:

    Returns:

    """

    def __init__(
        self,
        experiment_name,
        centroids_dir: Optional[str] = None,
        # meta_filepath="/Users/justinshenk/neurodata/data/Stroke_olive_oil/DVC cageids HT Maximilian Wiesmann updated.xlsx",
        meta_filepath="/Users/justinshenk/neurodata/data/mouse_ref.csv",
        cage_xmax=0.058 * 2,
        cage_ymax=0.125 * 2,
    ):
        # TODO: Fix in prod version
        self.basedir = "/Users/justinshenk/neurodata/"
        self._cpu_count = mp.cpu_count()
        self.centroids_dir = centroids_dir
        self.mouse_lookup = self.load_meta(meta_filepath)
        self.cage_xmax = cage_xmax
        self.cage_ymax = cage_ymax
        self.experiment_name = experiment_name
        self._init()
        self.outdir = os.path.join(
            self.basedir, "output", self._str2filename(experiment_name)
        )
        self.cages = self.get_cages()

    def _init(self):
        """ """
        if self.centroids_dir:
            search_path = glob.glob(os.path.join(self.centroids_dir, "*"))
            self.centroid_files = sorted(
                [
                    x.split("/")[-1]
                    for x in search_path
                    if "csv" in x and "filelist" not in x
                ]
            )
        else:
            self.centroid_files = None

        # plt.rc("font", family="serif")
        style.use("ggplot")
        sns.set_context("paper")

    @staticmethod
    def _str2filename(string):
        """

        Args:
          string: 

        Returns:

        """
        filename = string.replace(" ", "_")
        # TODO: Implement filename security
        filename = filename.replace("/", "")
        return filename

    def get_weekly_activity(self):
        """ """
        activity = self.get_daily_activity(days=33)
        weekly_list = []

        for week in range(-3, 5):
            for group in activity["Group+Diet"].unique():
                for period in [LIGHT, DARK]:
                    df = (
                        activity[
                            (
                                activity.Days_from_surgery >= week * 7 + 1
                            )  # ...-6, 1, 8, 15...
                            & (
                                activity.Days_from_surgery < (week + 1) * 7 + 1
                            )  # ...1, 8, 15, 21...
                            & (activity["Group+Diet"] == group)
                            & (activity.period == period)
                        ]
                        .groupby(["Cage"])
                        .Activity.mean()
                        .to_frame()
                    )
                    df["Group+Diet"] = group
                    df["Week"] = week
                    df["Period"] = period
                    # df['Cohort'] = [get_cohort(x) for x in df.index]
                    weekly_list.append(df)
        weekly = pd.concat(weekly_list)
        return weekly

    def ttests(
        self, daily: pd.DataFrame, days: List[int], day_pairs: List[tuple]
    ) -> list:
        """Returns pvalues for intra and interday variances."""
        pvalues = []

        # Paired Student's t-Test, between days for each group
        for day1, day2 in day_pairs:
            diets = sorted(daily.diet.unique())
            for period in daily.period.unique():
                day1_control = daily.loc[
                    (daily["diet"] == diets[0])
                    & (daily["days_from_surgery"] == day1)
                    & (daily.period == period),
                    "displacement",
                ]
                day2_control = daily.loc[
                    (daily["diet"] == diets[0])
                    & (daily["days_from_surgery"] == day2)
                    & (daily.period == period),
                    "displacement",
                ]
                day1_fortasyn = daily.loc[
                    (daily["diet"] == diets[1])
                    & (daily["days_from_surgery"] == day1)
                    & (daily.period == period),
                    "displacement",
                ]
                day2_fortasyn = daily.loc[
                    (daily["diet"] == diets[1])
                    & (daily["days_from_surgery"] == day2)
                    & (daily.period == period),
                    "displacement",
                ]

                if day1_control.shape == day2_control.shape:
                    try:
                        control_test = (
                            ttest_rel(day1_control, day2_control).pvalue,
                            (day1, day2, "Control"),
                            period,
                        )
                    except AttributeError:
                        print(f"Error: Empty data, skipping Days: {day1}, {day2}")
                        continue
                    pvalues.append(control_test)
                else:
                    print(
                        f"Control ({period}) - Skipping stats - Day {day1} shape is {len(day1_control)} and day {day2} shape is {len(day2_control)}"
                    )
                if day1_fortasyn.shape == day2_fortasyn.shape:
                    try:
                        fortasyn_test = (
                            ttest_rel(day1_fortasyn, day2_fortasyn).pvalue,
                            (day1, day2, "Fortasyn"),
                            period,
                        )
                    except AttributeError:
                        print("Error: Empty data, skipping Days: {day1}, {day2}")
                        continue
                    pvalues.append(fortasyn_test)
                else:
                    print(
                        f"Fortasyn ({period}) - Skipping stats - Day {day1} shape is {len(day1_fortasyn)} and day {day2} shape is {len(day2_fortasyn)}"
                    )

        # Get intraday significance
        for day in daily.days_from_surgery.unique():
            diets = sorted(daily.diet.unique())

            # Student's t-Test, between groups for given day
            for period in daily.period.unique():
                control = daily.loc[
                    (daily["diet"] == diets[0])
                    & (daily["days_from_surgery"] == day)
                    & (daily.period == period),
                    "displacement",
                ]
                fortasyn = daily.loc[
                    (daily["diet"] == diets[1])
                    & (daily["days_from_surgery"] == day)
                    & (daily.period == period),
                    "displacement",
                ]
                pvalues.append((ttest_ind(control, fortasyn).pvalue, day, period))
        return pvalues

    def plot_daily_distance(
        self,
        daily: Optional[pd.DataFrame] = None,
        days: list = [3],
        metric: str = "distance",
        day_pairs: list = [],
        **kwargs,
    ):
        """Plot daily distance.

        Args:

           daily:   dataframe with daily data
           days:    cutoff for plots, eg [3,7] plots with maximum 3 days and 7 days.

        """
        if daily is None:
            daily = self.get_distance(time="D")

        daily = daily.copy()

        # Get 35-day daily velocity
        cages = sorted(daily.cage.unique())
        if not "days_from_surgery" in daily:  # days_from_stroke provided
            raise Exception("`days_from_surgery` not in dataframe")

        diet_names = ["Control", "Fortasyn"]
        if sorted(daily.diet.unique().tolist()) != sorted(diet_names):
            for ind, diet in enumerate(sorted(daily.diet.unique())):
                daily["diet"] = daily.diet.replace(diet, diet_names[ind])
                print(f"Renamed diet column value {diet} to {diet_names[ind]}")

        # Save original name for diet column
        daily_diet_orig = daily.diet
        pvalues = self.ttests(daily, days, day_pairs)

        # Calculate and mark significant difference
        for max_days in days:
            fig, ax = plt.subplots(figsize=(14, 8))
            plt.title(f"Average daily {metric} {max_days} days after stroke")

            if kwargs.get("t_test"):
                self.plot_t_test(daily, pvalues, max_days)

            daily.diet = daily.diet.str.cat(" - " + daily.period)
            ax = sns.pointplot(
                x="days_from_surgery",
                y="displacement",
                data=daily.loc[(daily["days_from_surgery"] <= max_days)],
                markers=["d", "s", "^", "x"],
                linestyles=["--", "--", "-", "-"],
                palette=["b", "b", "r", "r"],
                hue="diet",
                hue_order=sorted(daily.diet.unique()),
                dodge=True,
                conf_lw=0.8,
                ci=68,  # SEM
            )

            plt.legend(loc="upper left")

            # Revert to original column
            daily.diet = daily_diet_orig

            plt.xlabel("Days from Surgery")
            if metric == "velocity":
                ylabel = "Velocity (cm/s)"
            elif metric == "distance":
                ylabel = "Distance Travelled (cm)"
            plt.ylabel(ylabel)

            plt.show()

    def plot_weekly(self, weekly: TrajaDataFrame, groups):
        """

        Args:
          weekly:
          groups: 

        Returns:

        """
        for group in groups:
            fig, ax = plt.subplots(figsize=(4, 3))
            for period in [LIGHT, DARK]:
                sns.pointplot(
                    x="Week",
                    y="Activity",
                    hue="Cohort",
                    data=weekly[
                        (weekly["Group+Diet"] == group) & (weekly["Period"] == period)
                    ]
                    .groupby("Activity")
                    .mean()
                    .reset_index(),
                    ci=68,
                )
            plt.title(group)
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(
                *sorted(zip(labels[:2], handles[:2]), key=lambda t: t[0])
            )
            ax.legend(handles, labels)
            # plt.tight_layout()
        plt.show()

    def get_presurgery_average_weekly_activity(self):
        """Average pre-stroke weeks into one point."""
        pre_average_weekly_act = os.path.join(self.outdir, "pre_average_weekly_act.csv")
        if not os.path.exists(pre_average_weekly_act):
            weekly = self.get_weekly_activity()
            for period in [LIGHT, DARK]:
                for cage in self.get_cages():
                    mean = weekly[
                        (weekly.index == cage)
                        & (weekly.Week < 0)
                        & (weekly.Period == period)
                    ].Activity.mean()
                    weekly.loc[
                        (weekly.index == cage)
                        & (weekly.Week < 0)
                        & (weekly.Period == period),
                        "Activity",
                    ] = mean
        else:
            weekly = self.read_csv(pre_average_weekly_act)
            return weekly

    def norm_weekly_activity(self, weekly):
        """

        Args:
          weekly: 

        Returns:

        """
        # Normalize activity
        weekly["Normed_Activity"] = 0
        for period in [LIGHT, DARK]:
            for cage in self.get_cages():
                df_night = weekly[
                    (weekly["Week"] >= -1)
                    & (weekly.index == cage)
                    & (weekly.Period == DARK)
                ]
                df = weekly[
                    (weekly["Week"] >= -1)
                    & (weekly.index == cage)
                    & (weekly.Period == period)
                ]
                assert df.Week.is_monotonic_increasing == True, "Not monotonic"
                normed = [x / df_night.Activity.values[0] for x in df.Activity.values]
                weekly.loc[
                    (weekly.index == cage)
                    & (weekly.Period == period)
                    & (weekly.Week >= -1),
                    "Normed_Activity",
                ] = normed
        return weekly

    def _stylize_axes(self, ax):
        """

        Args:
          ax: 

        Returns:

        """
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.xaxis.set_tick_params(top="off", direction="out", width=1)
        ax.yaxis.set_tick_params(right="off", direction="out", width=1)

    def _shift_xtick_labels(self, xtick_labels, first_index=None):
        """

        Args:
          xtick_labels: param first_index:  (Default value = None)
          first_index:  (Default value = None)

        Returns:

        """
        for idx, x in enumerate(xtick_labels):
            label = x.get_text()
            xtick_labels[idx].set_text(str(int(label) + 1))
            if first_index is not None:
                xtick_labels[0] = first_index
        return xtick_labels

    def _norm_daily_activity(self, activity):
        """

        Args:
          activity: 

        Returns:

        """
        norm_daily_activity_csv = os.path.join(self.outdir, "norm_daily_activity.csv")

        if not os.path.exists(norm_daily_activity_csv):
            activity["Normed_Activity"] = 0
            for period in [LIGHT, DARK]:
                for cage in self.get_cages():
                    # Get prestroke
                    prestroke_night_average = activity[
                        (activity.Days <= -1)
                        & (activity.cage == cage)
                        & (activity.Period == DARK)
                    ].Activity.mean()
                    df = activity[
                        (activity.Days >= -1)
                        & (activity.cage == cage)
                        & (activity.Period == period)
                    ]
                    assert (
                        df.Days_from_surgery.is_monotonic_increasing == True
                    ), "Not monotonic"
                    mean = activity[
                        (activity.Days <= -1)
                        & (activity.cage == cage)
                        & (activity.Period == period)
                    ].Activity.mean()
                    df.loc[
                        (df.Cage == cage) & (df.Period == period) & (df.Days == -1),
                        "Activity",
                    ] = mean
                    normed = [x / prestroke_night_average for x in df.Activity.values]
                    activity.loc[
                        (activity.Cage == cage)
                        & (activity.Period == period)
                        & (activity.Days >= -1),
                        "Normed_Activity",
                    ] = normed
            activity.to_csv(norm_daily_activity_csv)
        else:
            activity = pd.read_csv(norm_daily_activity_csv)
        return activity

    def plot_daily_normed_activity(self):
        """ """
        activity = self.get_daily_activity(days=33)
        normed_activity = self._norm_daily_activity(activity)
        ...

    def plot_weekly_normed_activity(self, presurgery_average=True):
        """Plot weekly normed activity. Optionally, average presurgery points.

        Args:
          presurgery_average: Default value = True)

        Returns:

        """
        if presurgery_average:
            weekly = self.get_presurgery_average_weekly_activity()
            # for cohort in [2,4]:
            fig, ax = plt.subplots(figsize=(6.25, 3.8))
            hue_order = weekly["Group+Diet"].unique()
            group_cnt = len(hue_order)
            for period in [LIGHT, DARK]:
                linestyles = (
                    ["--"] * group_cnt if period is LIGHT else ["-"] * group_cnt
                )
                sns.pointplot(
                    x="Week",
                    y="Normed_Activity",
                    hue="Group+Diet",
                    data=weekly[(weekly.Week >= -1) & (weekly.Period == period)],
                    #                                                                               (weekly.Cohort==cohort)],
                    palette=["k", "gray", "C0", "C1"][:group_cnt],
                    linestyles=linestyles,
                    # hue_order=['Sham - Control', 'Sham - HT', 'Stroke - Control', 'Stroke - HT'],
                    hue_order=hue_order,
                    markers=["d", "s", "^", "x"][
                        :group_cnt
                    ],  # TODO: Generalize for larger sets
                    dodge=True,
                    ci=68,
                )
            ax.set_xlabel("Weeks from Surgery")
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(
                *sorted(zip(labels[:4], handles[:4]), key=lambda t: t[0])
            )
            ax.legend(handles, labels)
            self._stylize_axes(ax)
            fig.set_facecolor("white")
            xtick_labels = ax.get_xticklabels()
            xtick_labels = self._shift_xtick_labels(xtick_labels, "Pre-surgery")

            plt.ylabel("Normalized Activity")
            ax.set_xticklabels(xtick_labels)
            plt.title("Normalized Activity")
            plt.show()

    def load_meta(self, meta_filepath):
        """

        Args:
          meta_filepath: 

        Returns:

        """
        if "xlsx" in meta_filepath:
            mouse_data = pd.read_excel(meta_filepath)[
                ["position", "Diet", "Sham_or_Stroke", "Stroke"]
            ]
            mouse_data["position"] = mouse_data["position"].apply(
                lambda x: x[1] + x[0].zfill(2)
            )
            return mouse_data.set_index("position").to_dict("index")
        elif "csv" in meta_filepath:
            mouse_ref = pd.read_csv("~/neurodata/data/mouse_ref.csv", sep="\t")
            mouse_ref = (
                mouse_ref.drop_duplicates().set_index("cage").drop(columns="mouse_nr")
            )

            return mouse_ref.to_dict()["diet"]

    def get_diet(self, cage):
        """

        Args:
          cage: 

        Returns:

        """
        return self.mouse_lookup[cage]["Diet"]

    def get_group(self, cage):
        """

        Args:
          cage: 

        Returns:

        """
        return self.mouse_lookup[cage]["Sham_or_Stroke"]

    def get_stroke(self, cage):
        """

        Args:
          cage: 

        Returns:

        """
        return self.mouse_lookup[cage]["Stroke"]

    def get_group_and_diet(self, cage):
        """

        Args:
          cage: 

        Returns:

        """
        diet = self.get_diet(cage)
        surgery = self.get_group(cage)
        return f"{'Sham' if surgery is 1 else 'Stroke'} - {'Control' if diet is 1 else 'HT'}"

    def get_cohort(self, cage):
        """

        Args:
          cage: 

        Returns:

        """
        # TODO: Generalize
        return self.mouse_lookup[cage]["Stroke"].month

    def get_cages(self):
        """ """
        return [x for x in self.mouse_lookup.keys()]

    def get_turn_ratios(self, file, angle_thresh, distance_thresh):
        """

        Args:
          file: param angle_thresh:
          distance_thresh: 
          angle_thresh: 

        Returns:

        """
        ratios = []
        cage = file.split("/")[-1].split("_")[0]
        # Get x,y coordinates from centroids
        date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S:%f")
        df = traja.read_file(file, index_col="time_stamps_vec")[["x", "y"]]
        # df.x = df.x.round(7)
        # df.y = df.y.round(7)
        df.traja.calc_distance()  # adds 'distance' column
        # TODO: Replace with generic intervention method name and lookup logic
        surgery_date = self.get_stroke(cage)
        df["Days_from_surgery"] = (df.index - surgery_date).days

        df.traja.calc_turn_angle()  # adds 'turn_angle' column
        #     df['turn_angle'].where((df['distance']>1e-3) & ((df.turn_angle > -15) & (df.turn_angle < 15))).hist(bins=30)
        #     df['turn_bias'] = df['turn_angle'] / .25 # 0.25s
        # Only look at distances over .01 meters, resample to minute intervals
        distance_mask = df["distance"] > (distance_thresh)
        angle_mask = ((df.turn_angle > angle_thresh) & (df.turn_angle < 90)) | (
            (df.turn_angle < -angle_thresh) & (df.turn_angle > -90)
        )

        day_mask = (df.index.hour >= 7) & (df.index.hour < 19)
        day_mean = df.loc[distance_mask & angle_mask & day_mask, "turn_angle"].dropna()
        night_mean = df.loc[
            distance_mask & angle_mask & ~day_mask, "turn_angle"
        ].dropna()
        right_turns_day = day_mean[day_mean > 0].shape[0]
        left_turns_day = day_mean[day_mean < 0].shape[0]
        right_turns_night = night_mean[night_mean > 0].shape[0]
        left_turns_night = night_mean[night_mean < 0].shape[0]
        ratios.append((df.Days_from_surgery[0], right_turns_day, left_turns_day, False))
        ratios.append(
            (df.Days_from_surgery[0], right_turns_night, left_turns_night, True)
        )

        ratios = [
            (day, right, left, period)
            for day, right, left, period in ratios
            if (left + right) > 0
        ]  # fix div by 0 error
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
        """

        Args:
          angle_thresh: Default value = 30)
          distance_thresh: Default value = 0.0025)

        Returns:

        """
        ratio_dict = {}
        for cage in self.get_cages():
            ratio_dict[cage] = []

            with mp.Pool(processes=self._cpu_count) as p:
                args = [
                    (file, angle_thresh, distance_thresh)
                    for file in self.centroid_files
                    if cage in file
                ]
                ratios = p.starmap(self.get_ratios, args)
                ratio_dict[cage].append(ratios)
                logging.info(f"Processed {cage}")

        turn_ratio_csv = os.path.join(
            self.outdir,
            f"ratios_angle-{angle_thresh}_distance-{distance_thresh}_period_turnangle.npy",
        )
        np.save(turn_ratio_csv, ratio_dict)
        logging.info(f"Saved to {turn_ratio_csv}")
        return ratio_dict

    def get_coords(self, cage):
        """

        Args:
          cage: 

        Returns:

        """
        path = os.path.join(self.outdir, "centroids", cage)
        df = traja.read_file(path)
        return df

    @staticmethod
    def _get_distance(file, dates, diet):
        try:
            time = "D"  # daily
            basename = os.path.basename(file)
            cage = basename.split("_")[0]
            date = "".join(basename.split("_raw_")[0].split("_")[1:])
            days_from_surgery = dates.index(date)
            df = traja.read_file(
                file, index_col="time_stamps_vec", parse_dates=["time_stamps_vec"]
            )
            displacement = df.traja.calc_displacement()
            displacement.dropna(inplace=True)
            # Resample
            displacement = displacement.to_frame()
            displacement["cage"] = cage
            displacement["diet"] = diet
            displacement["days_from_surgery"] = days_from_surgery

            # Day
            day = (
                displacement.between_time("7:00", "19:00")
                .resample(time)
                .agg({"displacement": "sum"})
            )
            day["cage"] = cage
            day["diet"] = diet
            day["days_from_surgery"] = days_from_surgery
            day["period"] = LIGHT

            # Night
            night = (
                displacement.between_time("19:00", "7:00", include_end=False)
                .resample(time)
                .agg({"displacement": "sum"})
            )
            night["cage"] = cage
            night["diet"] = diet
            night["days_from_surgery"] = days_from_surgery
            night["period"] = DARK

            # Overall
            overall = displacement.resample(time).agg({"displacement": "sum"})
            overall["cage"] = cage
            overall["diet"] = diet
            overall["days_from_surgery"] = days_from_surgery
            overall["period"] = "Overall"
            return [day, night, overall]
        except Exception as e:
            print(e, file)

    def get_distance(self, time="D"):
        """Get distance travelled resampled by time.

        Args:
            time (str): 'D' is day, 'H' hour, 'T' minutes, 'S' seconds, and so on.

        Returns:

        """
        df_list = []

        time = time.lower()
        if time is "d" and hasattr(self, "daily_distance"):
            return self.daily_distance
        elif time == "d" and not hasattr(self, "daily_distance"):
            print(f"Calculating daily distance for {self.cages}")
            self.distance_files_processed = 0
            for cage in sorted(self.cages):
                cage_files = sorted([x for x in self.centroid_files if cage in x])
                dates = [
                    "".join(x.split("_raw_")[0].split("_")[1:]) for x in cage_files
                ]
                pool = mp.Pool(processes=self._cpu_count - 1)
                args = [
                    (
                        os.path.join(self.centroids_dir, file),
                        dates,
                        self.mouse_lookup[cage],
                    )
                    for file in cage_files
                ]
                results = pool.starmap(self._get_distance, iterable=args)
                pool.close()
                pool.join()

                if isinstance(results, list):
                    for dfs in results:
                        if isinstance(dfs, list):
                            try:
                                df_list.extend(dfs)
                            except TypeError:
                                pass
                            except Exception as e:
                                print(e)
                    print(
                        f"[{len(df_list)//3}/{len(self.centroid_files)}] - Calculated distance for {cage}"
                    )
                else:
                    print(f"Results are None for {cage}")

        try:
            daily_distance = pd.concat(df_list, ignore_index=True)
        except Exception as e:
            print(e)
            return None
        self.daily_distance = daily_distance
        return self.daily_distance

    def plot_position_heatmap(self, cage, bins=20):
        """

        Args:
          cage: param bins:  (Default value = 20)
          bins:  (Default value = 20)

        Returns:

        """
        from numpy import unravel_index

        # TODO: Generate from y in +-0.12, x in +-0.058
        try:
            x0, x1 = self._trj.xlim
            y0, y1 = self._trj.ylim
        except:
            raise NotImplementedError("Not yet implemented automated heatmap binning")
        x_edges = np.linspace(x0, x1, num=bins)
        y_edges = np.linspace(y0, y1, num=bins)

        trj = self.get_coords(cage)
        x, y = zip(*trj[["x", "y"]].values)
        # TODO: Remove redundant histogram calculation
        H, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))
        cmax = H.flatten().argsort()[-2]  # Peak point is too hot, bug?

        fig, ax = plt.subplots()
        hist, x_edges, y_edges, image = ax.hist2d(
            np.array(y),
            np.array(x),
            bins=[
                np.linspace(trj.y.min(), trj.y.max(), 50),
                np.linspace(trj.x.min(), trj.x.max(), 50),
            ],
            cmax=cmax,
        )
        ax.colorbar()
        ax.set_aspect("equal")
        plt.show()
        # peak_index = unravel_index(hist.argmax(),hist.shape)

    def get_activity_files(self):
        """ """
        activity_dir = os.path.join(
            self.basedir, "data", self.experiment_name, "dvc_activation", "*"
        )
        activity_files = glob.glob(activity_dir)
        assert activity_files, "No activity files"
        return activity_files

    def aggregate_files(self):
        """Aggregate cage files into csvs"""
        os.makedirs(os.path.join(self.outdir, "centroids"), exist_ok=True)
        for cage in self.centroid_files:
            logging.info(f"Processing {cage}")
            # Check for aggregated cage file (eg, 'A04.csv')
            cage_path = os.path.join(self.outdir, "centroids", f"{cage}.csv")
            if os.path.exists(cage_path):
                continue
            # Otherwise, generate one
            search_path = os.path.join(self.centroids_dir, cage, "*.csv")
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
            cage_path = os.path.join(self.outdir, "centroids", f"{cage}.csv")
            df.to_csv(cage_path)
            logging.info(f"saved to {cage_path}")
        # activity_df = self.read_csv('data/Stroke_olive_oil/dvc_activation/A04.csv', index_col='time_stamp_start')
        return

    def _get_ratio_dict(self, angle=30, distance=0.0025):
        """

        Args:
          angle: Default value = 30)
          distance: Default value = 0.0025)

        Returns:

        """
        npy_path = os.path.join(
            self.outdir, "ratios_angle-{angle}_distance-{distance}_period_turnangle.npy"
        )
        r = np.load(npy_path)
        ratio_dict = r.item(0)
        return ratio_dict

    def get_cage_laterality(self, cage):
        """

        Args:
          cage: 

        Returns:

        """
        ratio_dict = self._get_ratio_dict()
        ratios = ratio_dict[cage]
        ratios = [x for x in ratios if (x[1] + x[2] > 0)]
        days = [day for day, _, _, nighttime in ratios if nighttime]

        laterality = [
            right_turns / (left_turns + right_turns)
            for day, right_turns, left_turns, nighttime in ratios
            if nighttime
        ]
        fig, ax = plt.subplots()
        ax.plot(days, laterality, label="Laterality")
        ax.set(
            title=f"{cage} laterality index (right/right+left)\nDistance threshold: 0.25 cm\nAngle threshold: {thresh}\nRight turn is > 0.5\n{self.get_diet(cage)}",
            xlabel="Days from surgery",
            ylabel="Laterality index",
        )
        ax.legend()
        ax.set_ylim((0, 1.0))
        ax2 = ax.twinx()
        ax2.plot(
            days,
            [right + left for _, right, left, nighttime in ratios if nighttime],
            color="C1",
            label="Number of turns",
        )
        ax2.set_ylabel("Number of turns")
        ax2.legend()
        plt.show()

    def get_minute_activity_files(
        self, excel_path="data/DVC_Activity_20180702.xlsx", day_limit=37
    ):
        minute_activity_paths = []

        # Look for preexisting files
        for cage in self.cages:
            cage_filepath = os.path.join(
                self.basedir,
                "data",
                self.experiment_name,
                cage.replace(" ", "_") + ".csv",
            )
            if os.path.exists(cage_filepath):
                minute_activity_paths.append(cage_filepath)
            else:
                print(f"Skipping {cage} because {cage_filepath} does not exist.")

        # If found, return
        if len(minute_activity_paths):
            return minute_activity_paths

        # Create [cage].csv files with activity resampled to minutes
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"{excel_path} not found.")
        xl = pd.ExcelFile(excel_path)
        r = re.compile("[A-Z]\d")
        activity_sheets = list(filter(r.match, xl.sheet_names))

        for sheet_name in activity_sheets:
            df = xl.parse(sheet_name)
            df = df.iloc[1:day_limit]  # remove post-37-day data
            df = df[df.columns[1:-76]]  # remove non activity columns
            cage_filepath = os.path.join(
                "data", self.experiment_name, sheet_name.replace(" ", "_") + ".csv"
            )
            minute_activity_paths.append(cage_filepath)
            df.to_csv(cage_filepath, index=False)
            print(f"Saving to {cage_filepath}")

        return minute_activity_paths

    def get_daily_activity(self, days: int) -> pd.DataFrame:
        """Returns daily activity."""
        minute_activity_files = self.get_minute_activity_files()

        # initialize
        activity = pd.DataFrame(columns=["cage", "activity", "Diet", "Days"])

        # get minute activity files
        sorted_cages = self.get_minute_activity_files()

        def csv_to_series(cage_file, fill_nan=True):
            cage = pd.read_csv(cage_file)
            if cage.shape[1] != 1439:
                print(
                    "Warning: Cage {cage} has {cage.shape[1]} columns instead of 1439"
                )
            nan_count = cage.isna().sum().sum()  # about 1440 * 8 NaNs
            if fill_nan:
                cage.fillna(0, inplace=True)
            flat = cage.values.flatten()
            cage_name = cage_file.split("/")[-1].split(".")[0]
            diet = self.mouse_lookup[cage_name]
            return flat, cage_name, diet

        for f in sorted_cages:
            flat, cage_name, diet = csv_to_series(f)
            day_indices = np.arange(390, 1110)
            mask = np.ones(1440, np.bool)
            mask[day_indices] = 0
            night_indices = np.arange(1440)[mask]
            for day in range(days):
                group = "Control" if int(diet) is 1 else "Fortasyn"
                overall_data = np.nanmean(flat[np.arange(1440) + day * 1440])
                day_data = np.nanmean(flat[day_indices + day * 1440])
                night_data = np.nanmean(flat[night_indices + day * 1440])
                activity = activity.append(
                    {
                        "activity": day_data,
                        "cage": cage_name,
                        "period": LIGHT,
                        "diet": group,
                        "Diet": group + f" - {LIGHT}",
                        "days": day + 1,
                    },
                    ignore_index=True,
                )
                activity = activity.append(
                    {
                        "activity": night_data,
                        "cage": cage_name,
                        "period": DARK,
                        "diet": group,
                        "Diet": group + f" - {DARK}",
                        "days": day + 1,
                    },
                    ignore_index=True,
                )
                activity = activity.append(
                    {
                        "activity": overall_data,
                        "cage": cage_name,
                        "period": "Overall",
                        "diet": group,
                        "Diet": group + " - Overall",
                        "days": day + 1,
                    },
                    ignore_index=True,
                )

            return activity

    # def get_daily_activity(self):
    #     """ """
    #     activity_csv = os.path.join(self.outdir, "daily_activity.csv")
    #     if not os.path.exists(activity_csv):
    #         print(f"Path {activity_csv} does not exist, creating dataframe")
    #         activity_list = []
    #         col_list = [f"e{i:02}" for i in range(1, 12 + 1)]  # electrode columns
    #
    #         # Iterate over minute activations
    #         search_path = os.path.join(
    #             self.basedir, "data", self.experiment_name, "dvc_activation", "*.csv"
    #         )
    #         minute_activity_files = sorted(glob.glob(search_path))
    #         for cage in minute_activity_files:
    #             cage_id = os.path.split(cage)[-1].split(".")[0]
    #             # TODO: Fix in final
    #             assert len(cage_id) == 3, logging.error(f"{cage_id} length != 3")
    #             # Read csv
    #             cage_df = pd.read_csv(
    #                 cage,
    #                 index_col="time_stamp_start",
    #                 date_parser=lambda x: pd.datetime.strptime(
    #                     x, "%Y-%m-%d %H:%M:%S:%f"
    #                 ),
    #             )
    #             # Make csv with columns for cage+activity+day+diet+surgery
    #             cage_df["Activity"] = cage_df[col_list].sum(axis=1)
    #             day = (
    #                 cage_df.traja.day.groupby(pd.Grouper(key="time", freq="D"))[
    #                     "Activity"
    #                 ]
    #                 .sum()
    #                 .to_frame()
    #             )
    #             day["Cage"] = cage_id
    #             day["Period"] = "LIGHT"
    #             day["Surgery"] = self.get_stroke(cage_id)
    #             day["Diet"] = self.get_diet(cage_id)
    #             day["Group"] = self.get_group(cage_id)
    #             day["Days"] = [int(x) for x in range(len(day.index))]
    #             activity_list.append(day)
    #
    #             night = (
    #                 cage_df.traja.night.groupby(pd.Grouper(key="time", freq="D"))[
    #                     "Activity"
    #                 ]
    #                 .sum()
    #                 .to_frame()
    #             )
    #             night["Cage"] = cage_id
    #             night["Period"] = "DARK"
    #             night["Surgery"] = self.get_stroke(cage_id)
    #             night["Diet"] = self.get_diet(cage_id)
    #             night["Group"] = self.get_group(cage_id)
    #             night["Days"] = [int(x) for x in range(len(night.index))]
    #             activity_list.append(night)
    #
    #         activity = pd.concat(activity_list)
    #         activity.to_csv(activity_csv)
    #     else:
    #         activity = traja.read_file(
    #             activity_csv,
    #             index_col="time_stamp_start",
    #             parse_dates=["Surgery", "time_stamp_start"],
    #             infer_datetime_format=True,
    #         )
    #     return activity

    def animate(self, trajectory, timesteps=None):
        """Animate trajectory over time with statistical information about turn angle, etc.

        Args:
          trajectory: param timesteps:  (Default value = None)
          timesteps:  (Default value = None)

        Returns:

        """
        if timesteps is not None:
            df = trajectory.iloc[:timesteps]
        else:
            df = trajectory
        ratios = {"left": 0, "right": 0}
        thresh = 30

        # Scale to centimeters (optional)
        df.x *= 100
        df.y *= 100
        if not "distance" in trajectory:
            df.traja.calc_distance()
        df.distance *= 100

        df.traja.calc_turn_angle()
        #     df['turn_angle'].where((df['distance']>1e-3) & ((df.turn_angle > -15) & (df.turn_angle < 15))).hist(bins=30)
        df["turn_bias"] = df["turn_angle"] / 0.25  # 0.25s
        # Only look at distances over .01 meters, resample to minute intervals
        distance_mask = df["distance"] > 1e-2
        angle_mask = (df.turn_bias > thresh) | (df.turn_bias < -thresh)

        # coords
        cage_y = 25.71
        cage_x = 13.7

        fig, axes = plt.subplots(
            2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [9, 1]}
        )

        def col_func(val, minval, maxval, startcolor, stopcolor):
            """Convert value in the range minval...maxval to a color in the range
                startcolor to stopcolor. The colors passed and the one returned are
                composed of a sequence of N component values (e.g. RGB).

            Args:
              val: param minval:
              maxval: param startcolor:
              stopcolor: 
              minval: 
              startcolor: 

            Returns:

            """
            f = float(val - minval) / (maxval - minval)
            return tuple(f * (b - a) + a for (a, b) in zip(startcolor, stopcolor))

        RED, YELLOW, GREEN = (1, 0, 0), (1, 1, 0), (0, 1, 0)
        CYAN, BLUE, MAGENTA = (0, 1, 1), (0, 0, 1), (1, 0, 1)
        steps = 10
        minval, maxval = 0.0, 1.0
        incr = (maxval - minval) / steps

        ax = axes[0]

        for i in df.index:
            ax.cla()
            ax.set_aspect("equal")
            ax.set_ylim(-cage_y / 2, cage_y / 2)
            ax.set_xlim(-cage_x / 2, cage_x / 2)

            x, y = df.loc[i, ["x", "y"]].values
            turn_bias = df.loc[i, "turn_bias"]

            # Scale to 0-1
            laterality = turn_bias + 360
            laterality /= 360 * 2
            if laterality > 1:
                laterality = 1
            elif laterality < 0:
                laterality = 0

            color = col_func(laterality, minval, maxval, BLUE, RED)
            ax.plot(x, y, color=color, marker="o")
            ax.invert_yaxis()

            try:
                # Filter for 1 cm/s
                # distance = df.distance.loc[i-2:i+2].sum()
                distance = df.distance.loc[i]
            except:
                print(f"Skipping {i}")
                continue

            count_turn = (distance >= 0.25) & ((turn_bias > 20) | (turn_bias < -20))
            if count_turn:
                if turn_bias > 0:
                    ratios["right"] += 1
                elif turn_bias < 0:
                    ratios["left"] += 1

            distance_str = (
                rf"$\bf{distance:.2f}$" if distance >= 0.25 else f"{distance:.2f}"
            )

            total_turns = ratios["right"] + ratios["left"]
            if total_turns == 0:
                overall_laterality = 0.5
            else:
                overall_laterality = ratios["right"] / total_turns

            ax.set_title(
                f"frame {i} - distance (cm/0.25s): {distance_str}\n \
                x: {x:.2f}, y: {y:.2f}\n \
                Heading: {df.loc[i,'heading']:5.0f} degrees\n \
                Turn Angle: {df.loc[i,'turn_angle']:4.0f}\n \
                Turn Bias: {turn_bias:4.0f}\n \
                Current Laterality: {laterality:.2f}\n \
                Left: {ratios['left']:>3}, Right: {ratios['right']:>3}\n \
                Overall Laterality: {overall_laterality:.2f} \
                "
            )

            axes[1].cla()
            if turn_bias > 0:
                rect = patches.Rectangle(
                    (0, 0), laterality, 1, linewidth=1, edgecolor="k", facecolor="r"
                )
            elif turn_bias < 0:
                rect = patches.Rectangle(
                    (0, 0), laterality, 1, linewidth=1, edgecolor="k", facecolor="b"
                )
            else:
                rect = patches.Rectangle(
                    (0, 0), laterality, 1, linewidth=1, edgecolor="k", facecolor="gray"
                )
            # Add the patch to the Axes
            axes[1].add_patch(rect)
            fig.tight_layout()
            plt.pause(0.01)

    def plot_t_test(self, daily, pvalues, max_days):
        # Plot asterisk for significance
        for idx, (pval, day, period) in enumerate(pvalues):
            if pval >= 0.05:
                continue
            if pval < 0.05:
                day_text = (
                    f"Day {day[0]} - Day {day[1]} - {day[2]}"
                    if isinstance(day, tuple)
                    else f"Day {day}"
                )
                print(f"{day_text} ({period}): {pval:.3f}")

            y_factor = (
                6
                if daily.loc[(daily.period == DARK), "displacement"].quantile(0.7) > 100
                else 2
            )

            y, h, col = (
                daily.loc[
                    (daily["days_from_surgery"] <= max_days) & (daily.period == period)
                ].displacement.quantile(0.75)
                + y_factor * idx,
                1.5,
                "k",
            )

            text = "*" if pval < 0.05 else ""
            text = "**" if pval < 0.01 else text
            text = "***" if pval < 0.001 else text

            day_range = range(daily.days_from_surgery.min(), max_days + 1)

            if isinstance(day, tuple):
                # Prevent plotting pvalues that won't fit on plot
                if day[1] > max_days:
                    continue

                # Annotate interday signfificance
                x1, x2 = (
                    np.searchsorted(day_range, day[0]),
                    np.searchsorted(day_range, day[1]),
                )

                diet = f"{day[2]} {period} p <= {pval:.3f}"
                plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                plt.text(
                    (x1 + x2) * 0.5,
                    y + h,
                    text + diet,
                    ha="center",
                    va="bottom",
                    color=col,
                    fontsize=14,
                )
            else:
                x1 = np.searchsorted(day_range, day)
                plt.text(
                    x1,
                    y + h,
                    text + f" p <= {pval:.3f}",
                    ha="center",
                    va="bottom",
                    color=col,
                    fontsize=14,
                )


def get_plot(file, axes, col=0):
    import traja

    df = traja.read_file(file, parse_dates=["time_stamps_vec"])
    for row, period in enumerate(["day", "night"]):
        if period is "day":
            period_df = df.traja.day()
        else:
            period_df = df.traja.night()
        filename = file.split("/")[-1].split(".")[0] + f"_{period}"
        traja.plotting.polar_bar(
            period_df,
            ax=axes[row, col],
            title=filename,
            save=filename + ".png",
            show=False,
        )


def debug():
    F4 = [
        "F6_2016-01-30_raw_centroid_positions.csv",
        "F4_2016-02-06_raw_centroid_positions.csv",
        "F4_2016-02-28_raw_centroid_positions.csv",
    ]
    F6 = [
        "F6_2016-01-30_raw_centroid_positions.csv",
        "F6_2016-02-28_raw_centroid_positions.csv",
    ]

    fig, axes = plt.subplots(2, 3, subplot_kw=dict(polar=True))

    for col, file in enumerate(F4):
        file = os.path.join("~/neurodata/data/raw_centroids_rev2", file)
        get_plot(file, axes, col)
    plt.show()


def pair(s: str):
    try:
        day_range = s.split("-")
        day_range = (int(day_range[0]), int(day_range[1]))
        if len(day_range) > 2:
            raise argparse.ArgumentTypeError(
                f"Day range length can only be 2, but it is {len(day_range)}"
            )
        return day_range
    except:
        raise argparse.ArgumentTypeError(
            "Day range must be hyphen separated, eg: --day-range=1-7"
        )


def pairs(s: str):
    try:
        day_pairs = s.split(",")
        day_pairs = [x.split("-") for x in day_pairs]
        day_pairs = [(int(x[0]), int(x[1])) for x in day_pairs]
        return day_pairs
    except:
        raise argparse.ArgumentTypeError(
            "Day pairs must be comma separated with hyphens for range, eg: --day-pairs=1-7,1-21"
        )


def days(s: str):
    try:
        days = [int(x) for x in s.split(",")]
        return days
    except:
        raise argparse.ArgumentTypeError(
            "Days must be comma separated, eg: --days=1,7,21"
        )


def main(args):
    daily_filename = f"daily_distance_{args.experiment_name}.csv"
    if os.path.exists(daily_filename):
        daily = pd.read_csv(daily_filename, dtype={"diet": "category"})
        print(f"Loaded {daily_filename}")
        experiment = DVCExperiment(experiment_name="fortasyn")
    elif args.centroids_dir and args.centroids_dir:
        centroids_dir = os.path.abspath(args.centroids_dir)
        if not os.path.exists(centroids_dir):
            raise Exception(f"Error: {centroids_dir} does not exist.")
        experiment = DVCExperiment(
            experiment_name="fortasyn", centroids_dir=centroids_dir
        )
        print(f"Computing daily distance.")
        daily = experiment.get_distance()
        daily = daily.drop_duplicates(
            daily, subset=["diet", "period", "cage", "days_from_surgery"]
        )
    else:
        raise Exception(
            f"Must provide centroids_dir or experiment_name, {daily_filename} not found"
        )
    if args.day_range:
        d1, d2 = args.day_range
        daily = daily.loc[
            (daily["days_from_surgery"] >= d1) & (daily["days_from_surgery"] <= d2)
        ]
    if args.day_pairs or args.days:
        print("Plotting daily distance")
        experiment.plot_daily_distance(daily, **args.__dict__)
    if args.activity:
        print("Plotting daily activity")
        activity = experiment.get_daily_activity(33)
        experiment.plot_daily_activity(activity)
    if args.glm:
        print("GLM:")
        glm(daily)


def parse_args():

    parser = argparse.ArgumentParser(description="Run DVC Experiment")
    parser.add_argument(
        "-c", "--centroids_dir", help="path to directory with centroids CSVs"
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="experiment_name",
        type=str,
        help="name of experiment, eg, fortasyn",
    )
    parser.add_argument(
        "-d",
        "--days",
        help="number of days per plot, comma-separated, eg: 3,7,35",
        type=days,
        default=[],
    )
    parser.add_argument(
        "-p",
        "--day_pairs",
        help="day pairs for statistical comparison, comma-separated, eg: 1-7,1-21",
        type=pairs,
        default=[],
    )
    parser.add_argument(
        "-g",
        "--glm",
        help="Estimate with generalized linear model",
        action="store_true",
    )
    parser.add_argument(
        "-r", "--day-range", help="Restrict day range, eg: 1-7", type=pair
    )
    parser.add_argument(
        "-t", "--t-test", help="Plot t-Test results", action="store_true"
    )
    parser.add_argument(
        "-a", "--activity", help="Plot daily activity", action="store_true"
    )
    args = parser.parse_args()

    max_days = 0
    if args.day_pairs:
        max_days = max(max(t) for t in args.day_pairs)
    if args.days:
        max_days = max(max_days, max(args.days))
    if args.day_range and args.days and args.day_range[-1] < max_days:
        raise Exception(
            f"Invalid options for day range: Day range {args.day_range}, Days {args.days}"
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
