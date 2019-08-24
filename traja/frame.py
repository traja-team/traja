import logging
from typing import Optional, List, Union, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import traja

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)


class TrajaDataFrame(pd.DataFrame):
    """A TrajaDataFrame object is a subclass of pandas :class:`<~pandas.dataframe.DataFrame>`.

    Args:
      args: Typical arguments for pandas.DataFrame.

    Returns:
      traja.TrajaDataFrame -- TrajaDataFrame constructor.
      
      .. doctest::

      >>> traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
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

    def copy(self, deep=True):
        """Make a copy of this TrajaDataFrame object

        Args:
          deep(bool, optional): Make a deep copy, i.e. also copy datasets (Default value = True)

        Returns:
          TrajaDataFrame -- copy

        """
        data = self._data
        if deep:
            data = data.copy()
        return TrajaDataFrame(data).__finalize__(self)

    def set(self, key, value):
        """Set metadata."""
        self.__dict__[key] = value


class TrajaCollection(object):
    """Collection of trajectories."""

    def __init__(
        self, trjs: Union[TrajaDataFrame, pd.DataFrame, dict], id_col: Optional[str]
    ):
        """Initialize with trajectories with x, y, and time columns.

        Args:
            trjs
            id_col

        """

        if isinstance(trjs, dict):
            trjs = []
            for name, df in trjs:
                df["id"] = name
                trjs.append(df)
            trjs = pd.concat(trjs)
        elif not id_col:
            raise Exception("id_col must be provided if trjs is not a dict")

        self.trjs = trjs

        self._id_col = id_col

    def plot(self, **kwargs):
        return traja.plotting.plot_collection(self.trjs, self._id_col, **kwargs)

    def apply_all(self, method, by="id", **kwargs):
        """Applies method to all trajectories"""
        return self.trjs.groupby(by=by).apply(method)


class StaticObject(object):
    def __init__(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        bounding_box: Tuple[float] = None,
    ):
        ...
        pass
