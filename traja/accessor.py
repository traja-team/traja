from collections import OrderedDict

import traja
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from shapely.geometry import shape


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

    def set(self, **kwargs):
        for key, value in kwargs:
            try:
                self._obj.__dict__[key] = value
            except Exception as e:
                raise Exception(f"Exception {e} assigning df.{key} to {value}")

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

    @property
    def night(self, begin="19:00", end="7:00"):
        """Get nighttime data between `begin` and `end`.

        Args:
          begin (str):  (Default value = '19:00')
          end (str):  (Default value = '7:00')

        Returns:
          trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory during night.

        """
        return self.between(begin, end)

    @property
    def day(self, begin="7:00", end="19:00"):
        """Get daytime data between `begin` and `end`.

        Args:
          begin (str):  (Default value = '7:00')
          end (str):  (Default value = '19:00')

        Returns:
          trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory during day.

        """
        return self.between(begin, end)

    def set(self, **kwargs):
        """Convenience function for setting metadata in the `traja` accessor."""
        for key, value in kwargs.items():
            try:
                self.__setattr__(key, value)
            except Exception as e:
                print(f"Cannot set {key} to {value}")

    def _get_plot_args(self, **kwargs):
        for var in self._obj._metadata:
            # Update global meta variables
            # TODO: Replace with elegant solution
            if var not in kwargs:
                # Metadata variable not specified in argument
                if var in self._obj.__dict__:
                    kwargs[var] = self._obj.__dict__[var]
        return kwargs

    def _get_time_col(self):
        """Returns time column in trajectory.

        Args:

        Returns:
           time_col (str or None): name of time column, 'index' or None

        """
        return traja.trajectory._get_time_col(self._obj)

    def between(self, begin, end):
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

    def resample_time(self, step_time):
        """Returns trajectory resampled with `step_time`.

        Args:
           step_time (float): Step time

        Returns:
            trj (:class:`~traja.frame.TrajaDataFrame`): Dataframe resampled.
        """
        return traja.trajectory.resample_time(self._obj, step_time=step_time)

    def trip_grid(
        self,
        bins=16,
        log=False,
        spatial_units=None,
        normalize=False,
        hist_only=False,
        plot=True,
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
        """Plot trajectory for single animal over period.

        Args:
          n_coords (int): Number of coordinates to plot
          **kwargs: additional keyword arguments to :meth:`matplotlib.axes.Axes.scatter`

        Returns:
            ax (:class:`~matplotlib.collections.PathCollection`): Axes of plot
        """
        ax = traja.plotting.plot(
            trj=self._obj,
            accessor=self,
            n_coords=n_coords,
            show_time=show_time,
            **kwargs,
        )
        return ax

    def _has_cols(self, cols: list):
        return traja.trajectory._has_cols(self._obj, cols)

    @property
    def xy(self, split=False):
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
            if split:
                xy = np.split(xy, [-1], axis=1)
            return xy
        else:
            raise Exception("'x' and 'y' are not in the dataframe.")

    def _check_has_time(self):
        """Check for presence of displacement time column."""
        time_col = self._get_time_col()
        if time_col is None:
            raise Exception("Missing time information in trajectory.")

    def calc_derivatives(self, assign=False):
        """Returns derivatives `displacement` and `displacement_time` as dictionary.

        Args:
          assign (bool): Assign output to ``TrajaDataFrame`` (Default value = False)

        Returns:
          derivs (:class:`~collections.OrderedDict`): Derivatives in dictionary.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3],'time':[0., 0.2, 0.4]})
            >>> df.traja.calc_derivatives()
            OrderedDict([('displacement', 0         NaN
            1    1.414214
            2    1.414214
            dtype: float64), ('displacement_time', 0    0.0
            1    0.2
            2    0.4
            Name: time, dtype: float64)])

        """
        derivs = traja.trajectory.calc_derivatives(self._obj)
        if assign:
            trj = self._obj.merge(
                pd.DataFrame.from_records(derivs), left_index=True, right_index=True
            )
            self._obj = trj
        return derivs

    def get_derivatives(self):
        """Get derivatives.

        Args:

        Returns:
          derivs (:class:`~collections.OrderedDict`) : Derivatives in dictionary.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3],'time':[0.,0.2,0.4]})
            >>> df.traja.get_derivatives()
            OrderedDict([('displacement', 0         NaN
            1    1.414214
            2    1.414214
            dtype: float64), ('displacement_time', 0    0.0
            1    0.2
            2    0.4
            Name: time, dtype: float64), ('speed', 0         NaN
            1    7.071068
            2    7.071068
            dtype: float64), ('speed_times', 1    0.2
            2    0.4
            Name: speed_times, dtype: float64), ('acceleration', 0    NaN
            1    NaN
            2    0.0
            dtype: float64), ('acceleration_times', 2    0.4
            Name: accleration_times, dtype: float64)])

        """
        if not self._has_cols(["displacement", "displacement_time"]):
            derivs = self.calc_derivatives(assign=False)
            d = derivs["displacement"]
            t = derivs["displacement_time"]
        else:
            d = self._obj.displacement
            t = self._obj.displacement_time
            derivs = OrderedDict(displacement=d, displacement_time=t)
        v = d[1 : len(d)] / t.diff()
        v.rename("speed")
        vt = t[1 : len(t)].rename("speed_times")
        # Calculate linear acceleration
        a = v.diff() / vt.diff().rename("acceleration")
        at = vt[1 : len(vt)].rename("accleration_times")
        data = OrderedDict(
            speed=v, speed_times=vt, acceleration=a, acceleration_times=at
        )
        derivs.update(data)
        return derivs

    @property
    def speed_intervals(
        self, faster_than=None, slower_than=None, interpolate_times=True
    ):
        """Calculate speed time intervals.
        
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
        derivs = self.get_derivatives()

        if faster_than is not None:
            pass
        if slower_than is not None:
            pass

        # Calculate trajectory speeds
        speed = derivs.get("speed")
        times = derivs.get("speed_times")
        flags = np.full(len(speed), 1)

        if faster_than is not None:
            flags = flags & (speed > faster_than)
        if slower_than is not None:
            flags = flags & (speed < slower_than)

        changes = np.diff(flags)
        stop_frames = np.where(changes == -1)[0]
        start_frames = np.where(changes == 1)[0]

        # Handle situation where interval begins or ends outside of trajectory
        if len(start_frames) > 0 or len(stop_frames) > 0:
            # Assume interval started at beginning of trajectory, since we don't know what happened before that
            if len(stop_frames) > 0 and (
                len(start_frames) == 0 or stop_frames[0] < start_frames[0]
            ):
                start_frames = np.append(1, start_frames)
            # Similarly, assume that interval can't extend past end of trajectory
            if (
                len(stop_frames) == 0
                or start_frames[len(start_frames) - 1]
                > stop_frames[len(stop_frames) - 1]
            ):
                stop_frames = np.append(stop_frames, len(speed))

        stop_times = times[stop_frames]
        start_times = times[start_frames]

        if interpolate_times and len(start_frames) > 0:
            # TODO: Implement
            raise NotImplementedError()
            r = self.linear_interp_times(
                slower_than, faster_than, speed, times, start_frames, start_times
            )
            start_times = r[:, 0]
            stop_times = r[:, 1]

        durations = stop_times - start_times
        result = traja.TrajaDataFrame(
            OrderedDict(
                start_frame=start_frames,
                start_time=start_times,
                stop_frame=stop_frames,
                stop_time=stop_times,
                duration=durations,
            )
        )

        metadata = OrderedDict(
            slower_than=slower_than,
            faster_than=faster_than,
            derivs=derivs,
            trajectory=self._obj,
        )
        result.__dict__.update(metadata)
        return result

    def to_shapely(self):
        """Return shapely object for area, bounds, etc. functions.

        Args:

        Returns:
          shapely.geometry.linestring.LineString -- Shapely shape.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> shape = df.traja.to_shapely()
            >>> shape.is_closed
            False

        """
        df = self._obj[["x", "y"]].dropna()
        coords = df.values
        tracks_obj = {"type": "LineString", "coordinates": coords}
        tracks_shape = shape(tracks_obj)
        return tracks_shape

    def calc_displacement(self, assign=True):
        """Returns ``Series`` of `float` with displacement between consecutive indices.

        Args:
          assign (bool, optional): Assign displacement to TrajaDataFrame (Default value = True)

        Returns:
          displacement (:class:`pandas.Series`): Displacement series.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.calc_displacement()
            0         NaN
            1    1.414214
            2    1.414214
            dtype: float64

        """
        displacement = traja.trajectory.calc_displacement(self._obj)
        if assign:
            self._obj = self._obj.assign(displacement=displacement)
        return displacement

    def calc_angle(self, assign=True):
        """Return ``Series`` with angle between steps as a function of displacement w.r.t x axis.

        Args:
          assign (bool, optional): Assign displacement to TrajaDataFrame (Default value = True)

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
        if not self._has_cols(["dx", "displacement"]):
            displacement = self.calc_displacement()
        else:
            displacement = self._obj.displacement

        angle = np.rad2deg(np.arccos(np.abs(self._obj.x.diff()) / displacement))
        if assign:
            self._obj["angle"] = angle
        return angle

    def scale(self, scale, spatial_units="m"):
        """Scale trajectory when converting, eg, from pixels to meters.

        Args:
          spatial_units(str., optional): Spatial units (eg, 'm') (Default value = "m")
          scale(float): Scale to convert coordinates

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

    def rediscretize(self, R):
        """Resample a trajectory to a constant step length. R is rediscretized step length.

        Args:
          R (float): Rediscretized step length (eg, 0.02)

        Returns:
          rt (:class:`numpy.ndarray`): rediscretized trajectory
          
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

    def calc_heading(self, assign=True):
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
        if not self._has_cols("angle"):
            angle = self.calc_angle(assign=True)
        else:
            angle = self._obj.angle
        df = self._obj
        dx = df.x.diff()
        dy = df.y.diff()
        # Get heading from angle
        mask = (dx > 0) & (dy >= 0)
        df.loc[mask, "heading"] = angle[mask]
        mask = (dx >= 0) & (dy < 0)
        df.loc[mask, "heading"] = -angle[mask]
        mask = (dx < 0) & (dy <= 0)
        df.loc[mask, "heading"] = -(180 - angle[mask])
        mask = (dx <= 0) & (dy > 0)
        df.loc[mask, "heading"] = 180 - angle[mask]
        if assign:
            self._obj["heading"] = df.heading
        return df.heading

    def calc_turn_angle(self, assign=True):
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
        if "heading" not in self._obj:
            heading = self.calc_heading(assign=False)
        else:
            heading = self._obj.heading
        turn_angle = heading.diff().rename("turn_angle")
        # Correction for 360-degree angle range
        turn_angle[turn_angle >= 180] -= 360
        turn_angle[turn_angle < -180] += 360
        if assign:
            self._obj["turn_angle"] = turn_angle
        return turn_angle
