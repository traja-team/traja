from typing import Union

import pandas as pd


def is_datetime_or_timedelta_dtype(series: Union[pd.Series, pd.Index]) -> bool:
    """Check whether pandas series or index is datetime or timedelta dtype.

    Args:
        series: Pandas Series or Index to check

    Returns:
        bool: True if series is datetime64 or timedelta64 dtype

    """
    return pd.api.types.is_datetime64_dtype(
        series
    ) or pd.api.types.is_timedelta64_dtype(series)
