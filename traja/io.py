import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_datetime_or_timedelta_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)

from traja import TrajaDataFrame


def read_file(
    filepath: str,
    id: str = None,
    parse_dates=False,
    xlim: tuple = None,
    ylim: tuple = None,
    spatial_units: str = "m",
    fps: float = None,
    **kwargs,
):
    """Convenience method wrapping pandas `read_csv` and initializing metadata.

    Args:
      filepath (str): path to csv file with `x`, `y` and `time` (optional) columns
      **kwargs: Additional arguments for :meth:`pandas.read_csv`.

    Returns:
        traj_df (:class:`~traja.main.TrajaDataFrame`): Trajectory

    """
    date_parser = kwargs.pop("date_parser", None)

    # TODO: Set index to first column containing 'time'
    df_test = pd.read_csv(
        filepath, nrows=10, parse_dates=parse_dates, infer_datetime_format=True
    )

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
