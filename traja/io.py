import numpy as np
import pandas as pd

from traja import TrajaDataFrame

def read_file(filepath:str, id=None, xlim=None, ylim=None, spatial_units='m', fps=None, **kwargs):
    """Convenience method wrapping pandas `read_csv` and initializing metadata.

    Args:
      filepath (str): path to csv file with `x`, `y` and `time` (optional) columns
      **kwargs: Additional arguments for :meth:`pandas.read_csv`.

    Returns:
        traj_df (:class:`~traja.main.TrajaDataFrame`): Trajectory

    """

    date_parser = kwargs.pop('data_parser', None)

    # TODO: Set index to first column containing 'time'
    df_test = pd.read_csv(filepath, nrows=10, parse_dates=True, infer_datetime_format=True)

    # Strip whitespace
    whitespace_cols = [c for c in df_test if ' ' in df_test[c].name]
    stripped_cols = {c: lambda x: x.strip() for c in whitespace_cols}
    converters = {**stripped_cols, **kwargs.pop('converters', {})}

    # Downcast to float32 # TODO: Benchmark float32 vs float64 for very big datasets
    float_cols = df_test.select_dtypes(include=[np.float]).columns
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
        elif fps is not None:
            time = np.array([x for x in trj.index], dtype=int) / fps
            trj['time'] = time
        else:
            # leave index as int frames
            pass
    else:
        # TODO: Implement for HDF5 and .npy files.
        raise NotImplementedError("Non-csv's not yet implemented")

    trj = TrajaDataFrame(trj)
    # Set meta properties of TrajaDataFrame
    trj.id = id
    trj.xlim = xlim
    trj.ylim = ylim
    trj.spatial_units = spatial_units
    trj.title = kwargs.get('title', None)
    trj.xlabel = kwargs.get('xlabel',None)
    trj.ylabel = kwargs.get('ylabel',None)
    trj.fps = fps
    return trj
