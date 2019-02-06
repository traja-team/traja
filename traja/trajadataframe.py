#! /usr/local/env python3
import logging
import sys

import traja
import matplotlib as mpl
if 'sphinx' in sys.argv[0]:
    mpl.use('agg')
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


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

    _metadata = ['xlim', 'ylim', 'spatial_units', 'xlabel', 'ylabel', 'title', 'fps', 'time_units', 'time_col', 'id']

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
        if method == 'merge':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == 'concat':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))

        # Requires 'time' column, so find one
        # time_col = self._get_time_col()
        # fps = kwargs.pop('fps',1)
        # assert isinstance(fps, (int, float)), f"{fps} is not a float or int"
        # if time_col is not None:
        #     self.rename(columns={time_col:'time'})
        # else:
        #     # Otherwise, create one from index and `fps`
        #     self['time'] = self.index
        #     self.time /= fps

        return self

    def _init_metadata(self):
        defaults = dict(fps=None,
                        spatial_units='m',
                        time_units='s')
        for name, value in defaults.items():
            if name not in self.__dict__:
                self.__dict__[name] = value

    def _get_time_col(self):
        time_cols = [col for col in self if 'time' in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            if is_numeric_dtype(self[time_col]):
                return time_col
        else:
            return None

    def copy(self, deep=True):
        """Make a copy of this TrajaDataFrame object

        Args:
          deep(bool, optional): Make a deep copy, i.e. also copy data (Default value = True)

        Returns:
          TrajaDataFrame -- copy

        """
        data = self._data
        if deep:
            data = data.copy()
        return TrajaDataFrame(data).__finalize__(self)


# TODO: Replace with tests.
# class Debug():
#     """Debug only.
#     """
#
#     def __init__(self, n_coords=1000):
#         import glob
#         import traja
#         from traja import TrajaAccessor
#         # files = glob.glob('/Users/justinshenk/neurodata/data/raw_centroids_rev2/*')
#         # df = traja.read_file(files[10])
#         # df.traja.set(xlim=(-0.06, 0.06),
#         #              ylim=(-0.13, 0.13),
#         #              xlabel=("x (m)"),
#         #              ylabel=("y (m)"),
#         #              title="Cage trajectory")
#         # FIXME: Function below takes forerver (or doesn't complete)
#         basepath = os.path.dirname(traja.__file__)
#         filepath = os.path.join(basepath, 'test', 'test_data', '3527.csv')
#         df = traja.read_file(filepath)
#         df.traja.plot()
