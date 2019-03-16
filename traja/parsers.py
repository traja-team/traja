from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_datetime_or_timedelta_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)

from traja import TrajaDataFrame


def from_df(df: pd.DataFrame, xcol=None, ycol=None, time_col=None, **kwargs):
    """Returns a :class:`traja.frame.TrajaDataFrame` from a :class:`pandas DataFrame<pandas.DataFrame>`.

    Args:
      df (:class:`pandas.DataFrame`): Trajectory as pandas ``DataFrame``
      xcol (str)
      ycol (str)
      timecol (str)

    Returns:
      traj_df (:class:`~traja.frame.TrajaDataFrame`): Trajectory

    .. doctest::

        >>> df = pd.DataFrame({'x':[0,1,2],'y':[1,2,3]})
        >>> traja.from_df(df)
           x  y
        0  0  1
        1  1  2
        2  2  3

    """
    traj_df = TrajaDataFrame(df)

    # Identify x and y columns if defined by user
    if xcol and ycol:
        traj_df["x"] = pd.to_numeric(traj_df[xcol], errors="coerce")
        traj_df["y"] = pd.to_numeric(traj_df[ycol], errors="coerce")
    if time_col:
        traj_df[time_col] = pd.to_timedelta(
            traj_df[time_col], unit=kwargs.get("time_units", "s")
        )
        kwargs.update({"time_col": time_col})

    # Initialize metadata
    for var in traj_df._metadata:
        if not hasattr(traj_df, var):
            traj_df.__dict__[var] = None

    # Save additional metadata
    for key, val in kwargs.items():
        traj_df.__dict__[key] = val
    return traj_df


def read_file(
    filepath: str,
    id: Optional[str] = None,
    xcol: Optional[str] = None,
    ycol: Optional[str] = None,
    parse_dates: Union[str, bool] = False,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    spatial_units: str = "m",
    fps: Optional[float] = None,
    **kwargs,
):
    """Convenience method wrapping pandas `read_csv` and initializing metadata.

    Args:
      filepath (str): path to csv file with `x`, `y` and `time` (optional) columns
      id (str): id for trajectory
      xcol (str): name of column containing x coordinates
      ycol (str): name of column containing y coordinates
      parse_dates (Union[list,bool]): The behavior is as follows:
                                    - boolean. if True -> try parsing the index.
                                    - list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3 each as a
                                    separate date column.
      xlim (tuple): x limits (min,max) for plotting
      ylim (tuple): y limits (min,max) for plotting
      spatial_units (str): for plotting (eg, 'cm')
      fps (float): for time calculations
      **kwargs: Additional arguments for :meth:`pandas.read_csv`.

    Returns:
        traj_df (:class:`~traja.main.TrajaDataFrame`): Trajectory

    """
    date_parser = kwargs.pop("date_parser", None)

    # TODO: Set index to first column containing 'time'
    df_test = pd.read_csv(
        filepath, nrows=10, parse_dates=parse_dates, infer_datetime_format=True
    )

    if xcol is not None or ycol is not None:
        if not xcol in df_test or ycol not in df_test:
            raise Exception(f"{xcol} or {ycol} not found as headers.")

    # Strip whitespace
    whitespace_cols = [c for c in df_test if " " in df_test[c].name]
    stripped_cols = {c: lambda x: x.strip() for c in whitespace_cols}
    converters = {**stripped_cols, **kwargs.pop("converters", {})}

    # Downcast to float32 # TODO: Benchmark float32 vs float64 for very big datasets
    float_cols = df_test.select_dtypes(include=[np.float]).columns
    float32_cols = {c: np.float32 for c in float_cols}

    # Convert string columns to categories
    string_cols = [c for c in df_test if df_test[c].dtype == str]
    category_cols = {c: "category" for c in string_cols}
    dtype = {**float32_cols, **category_cols, **kwargs.pop("dtype", {})}

    # Parse time column if present
    time_cols = [col for col in df_test.columns if "time" in col.lower()]
    time_col = time_cols[0] if time_cols else None

    if parse_dates and not date_parser and time_col:
        # try different parsers
        format_strs = [
            "%Y-%m-%d %H:%M:%S:%f",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ]
        for format_str in format_strs:
            date_parser = lambda x: pd.datetime.strptime(x, format_str)
            try:
                df_test = pd.read_csv(
                    filepath, date_parser=date_parser, nrows=10, parse_dates=[time_col]
                )
            except ValueError:
                pass
            if is_datetime64_any_dtype(df_test[time_col]):
                break
            elif is_timedelta64_dtype(df_test[time_col]):
                break
            else:
                # No datetime or timestamp column found
                date_parser = None

    if "csv" in filepath:
        trj = pd.read_csv(
            filepath,
            date_parser=date_parser,
            parse_dates=parse_dates or [time_col] if date_parser else False,
            converters=converters,
            dtype=dtype,
            **kwargs,
        )

        # TODO: Replace default column renaming with user option if needed
        if time_col:
            trj.rename(columns={time_col: "time"})
        elif fps is not None:
            time = np.array([x for x in trj.index], dtype=int) / fps
            trj["time"] = time
        else:
            # leave index as int frames
            pass
        if xcol and ycol:
            trj.rename(columns={xcol: "x", ycol: "y"})
    else:
        # TODO: Implement for HDF5 and .npy files.
        raise NotImplementedError("Non-csv's not yet implemented")

    trj = TrajaDataFrame(trj)

    # Set meta properties of TrajaDataFrame
    metadata = dict(
        id=id,
        xlim=xlim,
        spatial_units=spatial_units,
        title=kwargs.get("title", None),
        xlabel=kwargs.get("xlabel", None),
        ylabel=kwargs.get("ylabel", None),
        fps=fps,
    )
    trj.__dict__.update(**metadata)
    return trj
