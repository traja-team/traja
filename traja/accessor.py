from typing import Union

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

import traja


@pd.api.extensions.register_dataframe_accessor("traja")
class TrajaAccessor(object):
    """Accessor for pandas DataFrame with trajectory-specific numerical and analytical functions.

    Access with `df.traja`."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _strip(self, text):
        try:
            return text.strip()
        except AttributeError:
            return pd.to_numeric(text, errors="coerce")

    @staticmethod
    def _validate(obj):
        if "x" not in obj.columns or "y" not in obj.columns:
            raise AttributeError("Must have 'x' and 'y'.")

    @property
    def center(self):
        """Return the center point of this trajectory."""
        x = self._obj.x
        y = self._obj.y
        return (float(x.mean()), float(y.mean()))

    def night(self, begin: str = "19:00", end: str = "7:00"):
        """Get nighttime data between `begin` and `end`.

        Args:
          begin (str):  (Default value = '19:00')
          end (str):  (Default value = '7:00')

        Returns:
          trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory during night.

        """
        return self.between(begin, end)

    def day(self, begin: str = "7:00", end: str = "19:00"):
        """Get daytime data between `begin` and `end`.

        Args:
          begin (str):  (Default value = '7:00')
          end (str):  (Default value = '19:00')

        Returns:
          trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory during day.

        """
        return self.between(begin, end)

    def _get_time_col(self):
        """Returns time column in trajectory.

        Args:

        Returns:
           time_col (str or None): name of time column, 'index' or None

        """
        return traja.trajectory._get_time_col(self._obj)

    def between(self, begin: str, end: str):
        """Returns trajectory between `begin` and end` if `time` column is `datetime64`.

        Args:
          begin (str): Beginning of time slice.
          end (str): End of time slice.

        Returns:
          trj (:class:`~traja.frame.TrajaDataFrame`): Dataframe between values.
          
        .. doctest ::

            >>> s = pd.to_datetime(pd.Series(['Jun 30 2000 12:00:01', 'Jun 30 2000 12:00:02', 'Jun 30 2000 12:00:03']))
            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3],'time':s})
            >>> df.traja.between('12:00:00','12:00:01')
                             time  x  y
            0 2000-06-30 12:00:01  0  1

        """
        time_col = self._get_time_col()
        if time_col is "index":
            return self._obj.between_time(begin, end)
        elif time_col and is_datetime64_any_dtype(self._obj[time_col]):
            # Backup index
            dt_index_col = self._obj.index.name
            # Set dt_index
            trj = self._obj.copy()
            trj.set_index(time_col, inplace=True)
            # Create slice of trajectory
            trj = trj.between_time(begin, end)
            # Restore index and return column
            if dt_index_col:
                trj.set_index(dt_index_col, inplace=True)
            else:
                trj.reset_index(inplace=True)
            return trj
        else:
            raise TypeError("Either time column or index must be datetime64")

    def resample_time(self, step_time: float):
        """Returns trajectory resampled with ``step_time``.

        Args:
           step_time (float): Step time

        Returns:
            trj (:class:`~traja.frame.TrajaDataFrame`): Dataframe resampled.
        """
        return traja.trajectory.resample_time(self._obj, step_time=step_time)

    def trip_grid(
        self,
        bins: Union[int, tuple] = 16,
        log: bool = False,
        spatial_units=None,
        normalize: bool = False,
        hist_only: bool = False,
        plot: bool = True,
    ):
        """Returns a 2D histogram of trip.

        Args:
          bins (int, optional): Number of bins (Default value = 16)
          log (bool): log scale histogram (Default value = False)
          spatial_units (str): units for plotting
          normalize (bool): normalize histogram into density plot
          hist_only (bool): return histogram without plotting

        Returns:
            hist (:class:`numpy.ndarray`): 2D histogram as array
            image (:class:`matplotlib.collections.PathCollection`: image of histogram

        """
        hist, image = traja.plotting.trip_grid(
            self._obj,
            bins=bins,
            log=log,
            spatial_units=self._obj.spatial_units,
            normalize=normalize,
            hist_only=hist_only,
            plot=plot,
        )
        return hist, image

    def plot(self, n_coords: int = None, show_time=False, **kwargs):
        """Plot trajectory over period.

        Args:
          n_coords (int): Number of coordinates to plot
          **kwargs: additional keyword arguments to :meth:`matplotlib.axes.Axes.scatter`

        Returns:
            fig (:class:`~matplotlib.figure.Figure`): Axes of plot
        """
        fig = traja.plotting.plot(
            trj=self._obj,
            accessor=self,
            n_coords=n_coords,
            show_time=show_time,
            **kwargs,
        )
        return fig

    def plot_flow(self, kind="quiver", **kwargs):
        """Plot grid cell flow.

        Args:
          kind (str): Kind of plot (eg, 'quiver','surface','contour','contourf','stream')
          **kwargs: additional keyword arguments to :meth:`matplotlib.axes.Axes.scatter`

        Returns:
            fig (:class:`~matplotlib.figure.Figure`): Axes of plot

        """
        fig = traja.plotting.plot_flow(trj=self._obj, kind=kind, **kwargs)
        return fig

    def _has_cols(self, cols: list):
        return traja.trajectory._has_cols(self._obj, cols)

    @property
    def xy(self):
        """Returns a :class:`numpy.ndarray` of x,y coordinates.

        Args:
            split (bool): Split into seaprate x and y :class:`numpy.ndarrays`

        Returns:
          xy (:class:`numpy.ndarray`) -- x,y coordinates (separate if `split` is `True`)
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.xy
            array([[0, 1],
                   [1, 2],
                   [2, 3]])

        """
        if self._has_cols(["x", "y"]):
            xy = self._obj[["x", "y"]].values
            return xy
        else:
            raise Exception("'x' and 'y' are not in the dataframe.")

    def _check_has_time(self):
        """Check for presence of displacement time column."""
        time_col = self._get_time_col()
        if time_col is None:
            raise Exception("Missing time information in trajectory.")

    def calc_derivatives(self, assign: bool = False):
        """Returns derivatives `displacement` and `displacement_time`.

        Args:
          assign (bool): Assign output to ``TrajaDataFrame`` (Default value = False)

        Returns:
          derivs (:class:`~collections.OrderedDict`): Derivatives.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3],'time':[0., 0.2, 0.4]})
            >>> df.traja.calc_derivatives()
               displacement  displacement_time
            0           NaN                0.0
            1      1.414214                0.2
            2      1.414214                0.4


        """
        derivs = traja.trajectory.calc_derivatives(self._obj)
        if assign:
            trj = self._obj.merge(derivs, left_index=True, right_index=True)
            self._obj = trj
        return derivs

    def get_derivatives(self) -> pd.DataFrame:
        """Returns derivatives as DataFrame."""
        derivs = traja.trajectory.get_derivatives(self._obj)
        return derivs

    @property
    def speed_intervals(
        self,
        faster_than: Union[float, int] = None,
        slower_than: Union[float, int] = None,
        interpolate_times: bool = True,
    ):
        """Returns ``TrajaDataFrame`` with speed time intervals.

        Returns a dictionary of time intervals where speed is slower and/or faster than specified values.

        Args:
          faster_than (float, optional): Minimum speed threshold. (Default value = None)
          slower_than (float or int, optional): Maximum speed threshold. (Default value = None)
          interpolate_times (bool, optional): Interpolate times between steps. (Default value = True)

        Returns:
          result (:class:`~collections.OrderedDict`) -- time intervals as dictionary.

        .. note::

            Implementation ported to Python, heavily inspired by Jim McLean's trajr package.

        """
        result = traja.trajectory.speed_intervals(
            self._obj, faster_than, slower_than, interpolate_times
        )
        return result

    def to_shapely(self):
        """Returns shapely object for area, bounds, etc. functions.

        Args:

        Returns:
          shape (shapely.geometry.linestring.LineString): Shapely shape.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> shape = df.traja.to_shapely()
            >>> shape.is_closed
            False

        """
        trj = self._obj[["x", "y"]].dropna()
        tracks_shape = traja.trajectory.to_shapely(trj)
        return tracks_shape

    def calc_displacement(self, assign: bool = True) -> pd.Series:
        """Returns ``Series`` of `float` with displacement between consecutive indices.

        Args:
          assign (bool, optional): Assign displacement to TrajaAccessor (Default value = True)

        Returns:
          displacement (:class:`pandas.Series`): Displacement series.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.calc_displacement()
            0         NaN
            1    1.414214
            2    1.414214
            Name: displacement, dtype: float64

        """
        displacement = traja.trajectory.calc_displacement(self._obj)
        if assign:
            self._obj = self._obj.assign(displacement=displacement)
        return displacement

    def calc_angle(self, assign: bool = True) -> pd.Series:
        """Returns ``Series`` with angle between steps as a function of displacement w.r.t x axis.

        Args:
          assign (bool, optional): Assign turn angle to TrajaAccessor (Default value = True)

        Returns:
          angle (:class:`pandas.Series`): Angle series.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.calc_angle()
            0     NaN
            1    45.0
            2    45.0
            dtype: float64

        """
        angle = traja.trajectory.calc_angle(self._obj)
        if assign:
            self._obj["angle"] = angle
        return angle

    def scale(self, scale: float, spatial_units: str = "m"):
        """Scale trajectory when converting, eg, from pixels to meters.

        Args:
          scale(float): Scale to convert coordinates
          spatial_units(str., optional): Spatial units (eg, 'm') (Default value = "m")

        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.scale(0.1)
            >>> df
                 x    y
            0  0.0  0.1
            1  0.1  0.2
            2  0.2  0.3

        """
        self._obj[["x", "y"]] *= scale
        self._obj.__dict__["spatial_units"] = spatial_units

    def _transfer_metavars(self, df):
        for attr in self._obj._metadata:
            df.__dict__[attr] = getattr(self._obj, attr, None)
        return df

    def rediscretize(self, R: float):
        """Resample a trajectory to a constant step length. R is rediscretized step length.

        Args:
          R (float): Rediscretized step length (eg, 0.02)

        Returns:
          rt (:class:`traja.TrajaDataFrame`): rediscretized trajectory
          
        .. note::
          
            Based on the appendix in Bovet and Benhamou, (1988) and Jim McLean's
            `trajr <https://github.com/JimMcL/trajr>`_ implementation.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.rediscretize(1.)
                      x         y
            0  0.000000  1.000000
            1  0.707107  1.707107
            2  1.414214  2.414214

        """
        rt = traja.trajectory.rediscretize_points(self._obj, R)
        self._transfer_metavars(rt)
        return rt

    def calc_heading(self, assign: bool = True):
        """Calculate trajectory heading.

        Args:
          assign (bool): (Default value = True)

        Returns:
            heading (:class:`pandas.Series`): heading as a ``Series``

        ..doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.calc_heading()
            0     NaN
            1    45.0
            2    45.0
            Name: heading, dtype: float64

        """
        heading = traja.trajectory.calc_heading(self._obj)
        if assign:
            self._obj["heading"] = heading
        return heading

    def calc_turn_angle(self, assign: bool = True):
        """Calculate turn angle.

        Args:
          assign (bool):  (Default value = True)

        Returns:
            turn_angle (:class:`~pandas.Series`): Turn angle

        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.calc_turn_angle()
            0    NaN
            1    NaN
            2    0.0
            Name: turn_angle, dtype: float64

        """
        turn_angle = traja.trajectory.calc_turn_angle(self._obj)

        if assign:
            self._obj["turn_angle"] = turn_angle
        return turn_angle
