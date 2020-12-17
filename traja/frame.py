import copy
import logging
from typing import Optional, List, Union, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype

import traja

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)


class TrajaDataFrame(pd.DataFrame):
    """A TrajaDataFrame object is a subclass of pandas :class:`<~pandas.dataframe.DataFrame>`.

    Args:
      args: Typical arguments for pandas.DataFrame.

    Returns:
      traja.TrajaDataFrame -- TrajaDataFrame constructor.
      
        >>> traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]}) # doctest: +SKIP
           x  y
        0  0  1
        1  1  2
        2  2  3

    """

    _metadata = [
        "xlim",
        "ylim",
        "spatial_units",
        "xlabel",
        "ylabel",
        "title",
        "fps",
        "time_units",
        "time_col",
        "id",
    ]

    def __init__(self, *args, **kwargs):
        # Allow setting metadata from constructor
        traja_kwargs = dict()
        for key in list(kwargs.keys()):
            for name in self._metadata:
                if key == name:
                    traja_kwargs[key] = kwargs.pop(key)
        super(TrajaDataFrame, self).__init__(*args, **kwargs)

        if len(args) == 1 and isinstance(args[0], TrajaDataFrame):
            args[0]._copy_attrs(self)
        for name, value in traja_kwargs.items():
            self.__dict__[name] = value

        # Initialize metadata like 'fps','spatial_units', etc.
        self._init_metadata()

    @property
    def _constructor(self):
        return TrajaDataFrame

    def _copy_attrs(self, df):
        for attr in self._metadata:
            df.__dict__[attr] = getattr(self, attr, None)

    def __finalize__(self, other, method=None, **kwargs):
        """propagate metadata from other to self """
        # merge operation: using metadata of the left object
        if method == "merge":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == "concat":
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        return self

    # def __getitem__(self, key):
    #     """
    #       If result is a DataFrame with a x or X column, return a
    #       TrajaDataFrame.
    #       """
    #     result = super(TrajaDataFrame, self).__getitem__(key)
    #     if isinstance(result, DataFrame) and "x" == result or "X" == result:
    #         result.__class__ = TrajaDataFrame
    #     elif isinstance(result, DataFrame):
    #         result.__class__ = DataFrame
    #     return result

    def _init_metadata(self):
        defaults = dict(fps=None, spatial_units="m", time_units="s")
        for name, value in defaults.items():
            if name not in self.__dict__:
                self.__dict__[name] = value

    def _get_time_col(self):
        time_cols = [col for col in self if "time" in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            if is_numeric_dtype(self[time_col]):
                return time_col
        else:
            return None

    @classmethod
    def from_xy(cls, xy: np.ndarray):
        """Convenience function for initializing :class:`~traja.frame.TrajaDataFrame` with x,y coordinates.

        Args:
            xy (:class:`numpy.ndarray`): x,y coordinates

        Returns:
            traj_df (:class:`~traja.frame.TrajaDataFrame`): Trajectory as dataframe

        .. doctest::

            >>> import numpy as np
            >>> xy = np.array([[0,1],[1,2],[2,3]])
            >>> traja.from_xy(xy)
               x  y
            0  0  1
            1  1  2
            2  2  3

        """
        df = cls.from_records(xy, columns=["x", "y"])
        return df

    def set(self, key, value):
        """Set metadata."""
        self.__dict__[key] = value


def tocontainer(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return TrajaCollection(result)

    return wrapper


class TrajaCollection(TrajaDataFrame):
    """Collection of trajectories."""

    _metadata = [
        "xlim",
        "ylim",
        "spatial_units",
        "xlabel",
        "ylabel",
        "title",
        "fps",
        "time_units",
        "time_col",
        "_id_col",
    ]

    def __init__(
        self,
        trjs: Union[TrajaDataFrame, pd.DataFrame, dict],
        id_col: Optional[str] = None,
        **kwargs,
    ):
        """Initialize with trajectories with x, y, and time columns.

        Args:self.
            trjs
            id_col (str) - Default is "id"

        """
        # Add id column
        if isinstance(trjs, dict):
            _trjs = []
            for name, df in trjs.items():
                df["id"] = name
                _trjs.append(df)
            super(TrajaCollection, self).__init__(pd.concat(_trjs), **kwargs)
        elif isinstance(trjs, (TrajaDataFrame, DataFrame)):
            super(TrajaCollection, self).__init__(trjs, **kwargs)
        else:
            super(TrajaCollection, self).__init__(trjs, **kwargs)

        if id_col:
            self._id_col = id_col
        elif hasattr(self, "_id_col"):
            self._id_col = self._id_col
        else:
            self._id_col = "id"  # default

    @property
    def _constructor(self):
        return TrajaCollection

    def _copy_attrs(self, df):
        for attr in self._metadata:
            df.__dict__[attr] = getattr(self, attr, None)

    # def __copy__(self):
    #     return TrajaCollection(self.trjs).__dict__.update(self.__dict__)

    def __repr__(self):
        return "TrajaCollection:\n" + super(TrajaCollection, self).__repr__()

    # def __add__(self, other):
    #     trjs = self.trjs.append(other, ignore_index=True)
    #     return TrajaCollection(trjs, id_col=self._id_col)

    def plot(self, colors=None, **kwargs):
        """Plot collection of trajectories with colors assigned to each id.

            >>> trjs = {ind: traja.generate(seed=ind) for ind in range(3)} # doctest: +SKIP
            >>> coll = traja.TrajaCollection(trjs) # doctest: +SKIP
            >>> coll.plot() # doctest: +SKIP

        """
        return traja.plotting.plot_collection(
            self, self._id_col, colors=colors, **kwargs
        )

    def apply_all(self, method, **kwargs):
        """Applies method to all trajectories

        Args:
            method

        Returns:
            dataframe or series

            >>> trjs = {ind: traja.generate(seed=ind) for ind in range(3)} # doctest: +SKIP
            >>> coll = traja.TrajaCollection(trjs) # doctest: +SKIP
            >>> angles = coll.apply_all(traja.calc_angles) # doctest: +SKIP

        """
        return self.groupby(by=self._id_col).apply(method)


class StaticObject(object):
    def __init__(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        bounding_box: Tuple[float] = None,
    ):
        ...
        pass
