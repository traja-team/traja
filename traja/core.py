import pandas as pd

# Check whether pandas series is datetime or timedelta
def is_datetime_or_timedelta_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_dtype(series) or pd.api.types.is_timedelta64_dtype(series)