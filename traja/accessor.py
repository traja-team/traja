#! /usr/local/env python3
import argparse
import glob
import logging
import os
import sys
from collections import OrderedDict

import traja
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

from scipy.spatial.distance import directed_hausdorff, euclidean
from numpy import unravel_index
from shapely.geometry import shape
from traja.utils import polar_to_z

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


@pd.api.extensions.register_dataframe_accessor("traja")
class TrajaAccessor(object):
    """Accessor for pandas DataFrame with trajectory-specific numerical and analytical functions."""

    def __init__(self, pandas_obj):
        self._trj = pandas_obj

    def _strip(self, text):
        try:
            return text.strip()
        except AttributeError:
            return pd.to_numeric(text, errors='coerce')

    def set(self, **kwargs):
        for key, value in kwargs:
            try:
                self._trj.__dict__[key] = value
            except Exception as e:
                raise Exception(f"Exception {e} assigning df.{key} to {value}")

    @property
    def night(self, begin='19:00', end='7:00'):
        """Returns trajectory indices for time between `begin` and `end`.

        Args:
          begin:  (Default value = '19:00')
          end:  (Default value = '7:00')

        Returns:

        """
        return self.between(begin, end)

    @property
    def day(self, begin='7:00', end='19:00'):
        """

        Args:
          begin:  (Default value = '7:00')
          end:  (Default value = '19:00')

        Returns:
          

        """
        return self.between(begin, end)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            try:
                self.__setattr__(key, value)
            except Exception as e:
                logging.ERROR(f"Cannot set {key} to {value}")

    def _get_plot_args(self, **kwargs):
        for var in self._trj._metadata:
            # Update global meta variables
            # TODO: Replace with elegant solution
            if var not in kwargs:
                # Metadata variable not specified in argument
                if var in self._trj.__dict__:
                    kwargs[var] = self._trj.__dict__[var]
        return kwargs

    def get_time_col(self):
        time_cols = [col for col in self._trj if 'time' in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            if is_numeric_dtype(self._trj[time_col]):
                return time_col
        else:
            return None

    def between(self, begin, end):
        """Return trajectory between `begin` and end` if `time` column is `datetime64`.

        Args:
          begin(str.): Beginning of time slice.
          end(str.): End of time slice.

        Returns:
          TrajaDataFrame -- Data frame between values.
          
        .. doctest ::

            >>> s = pd.to_datetime(pd.Series(['Jun 30 2000 12:00:01', 'Jun 30 2000 12:00:02', 'Jun 30 2000 12:00:03']))
            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3],'time':s})
            >>> df.traja.between('12:00:00','12:00:01')
                             time  x  y
            0 2000-06-30 12:00:01  0  1

        """
        if pd.core.dtypes.common.is_datetime64_dtype(self._trj.time):
            self._trj.set_index('time', inplace=True)
            df = self._trj.between_time(begin, end)
            df = df.reset_index()
            return df
        else:
            raise TypeError(f"{self._trj.time.dtype} must be datetime64")

    def plot(self, n_coords: int = None, **kwargs):
        """Plot trajectory for single animal over period.

        Args:
          n_coords(int.): Number of coordinates to plot
          n_coords: int:  (Default value = None)
          **kwargs: 

        Returns:

        """
        GRAY = '#999999'

        kwargs = self._get_plot_args(**kwargs)
        xlim = kwargs.pop('xlim', None)
        ylim = kwargs.pop('ylim', None)
        xlabel = kwargs.pop('xlabel', f'x ({self._trj.spatial_units})')
        ylabel = kwargs.pop('ylabel', f'y ({self._trj.spatial_units})')
        title = kwargs.pop('title', None)
        time_units = kwargs.pop('time_units', None)
        fps = kwargs.pop('fps', None)

        if n_coords is not None:
            raise NotImplementedError("Days and n_coords cannot both be specified.")

        start, end = None, None
        coords = self._trj[['x', 'y']]
        time_col = self.get_time_col()

        if n_coords is not None:
            # Plot first `n_coords`
            start, end = 0, n_coords
            verts = coords.iloc[:n_coords].values
        else:
            start, end = 0, len(coords)
            verts = coords.iloc[:end].values

        n_coords = len(verts)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        path = Path(verts, codes)

        fig, ax = plt.subplots()
        patch = patches.PathPatch(path, edgecolor=GRAY, facecolor='none', lw=3, alpha=0.3)
        ax.add_patch(patch)

        xs, ys = zip(*verts)

        colors = plt.cm.viridis(np.linspace(0, 1, n_coords))
        ax.scatter(xs, ys, c=colors, s=8, zorder=2, alpha=0.3)

        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim((coords.x.min(), coords.x.max()))
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim((coords.y.min(), coords.y.max()))

        if kwargs.pop('invert_yaxis', None):
            plt.gca().invert_yaxis()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect('equal')

        N = 21  # bins
        cmap = plt.get_cmap('viridis', N)
        norm = mpl.colors.Normalize(vmin=0, vmax=N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = plt.colorbar(sm)
        cbar_yticklabels = cbar.ax.get_yticklabels()
        interval = n_coords // len(cbar_yticklabels)
        if time_col:
            cbar_yticklabels = [self._trj[time_col][interval * i] for i in range(len(cbar_yticklabels))]
        else:
            cbar_yticklabels = [coords.index[interval * i] for i in range(len(cbar_yticklabels))]
        cbar.ax.set_yticklabels(cbar_yticklabels)
        # if time_col or time_units:
        #     cbar.ax.set_ylabel(f'{time_units}')
        plt.tight_layout()
        plt.show()
        return ax

    # def polar_bar(self):

    def trip_grid(self, bins=16, log=False):
        """Generate a heatmap of time spent by point-to-cell gridding.

        Args:
          bins(int., optional): Number of bins (Default value = 16)
          log:  (Default value = False)

        Returns:

        """
        # TODO: Add kde-based method for line-to-cell gridding
        df = self._trj[['x', 'y']].dropna()
        x0, x1 = df.xlim or (df.x.min(), df.x.max())
        y0, y1 = df.ylim or (df.y.min(), df.y.max())
        aspect = (y1 - y0) / (x1 - x0)
        x_edges = np.linspace(x0, x1, num=bins)
        y_edges = np.linspace(y0, y1, num=int(bins / aspect))

        x, y = zip(*df.values)
        # # TODO: Remove redundant histogram calculation
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges))
        fig, ax = plt.subplots()
        if log:
            hist = np.log(hist + np.e)
        image = plt.imshow(hist, interpolation='bilinear')
        # TODO: Set xticks and yticks to original data coordinates
        # TODO: Adjust colorbar ytick_labels to correspond with time
        cbar = plt.colorbar(image, ax=ax)
        plt.title("Time spent{}".format(' (Logarithmic)' if log else ''))
        plt.tight_layout()
        plt.show()
        # TODO: Add most common locations in grid
        # peak_index = unravel_index(hist.argmax(), hist.shape)

    def _has_cols(self, cols: list):
        return set(cols).issubset(self._trj.columns)

    @property
    def xy(self):
        """Return numpy array of x,y coordinates.

        Args:

        Returns:
          np.ndarray -- x,y coordinates
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.xy
            array([[0, 1],
                   [1, 2],
                   [2, 3]])

        """
        if self._has_cols(['x', 'y']):
            xy = self._trj[['x', 'y']].values
            return xy
        else:
            raise Exception("'x' and 'y' are not in the dataframe.")

    def _check_has_time(self):
        """Check for presence of displacement time column."""
        if 'time' not in self._trj:
            raise Exception("Missing time information in trajectory.")

    def calc_derivatives(self, assign=False):
        """Calculate derivatives `displacement` and `displacement_time`.

        Args:
          assign (bool): Assign output to `TrajaDataFrame` (Default value = False)

        Returns:
          dict: Derivatives in dictionary.
          
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
        self._check_has_time()
        if not 'displacement' in self._trj:
            displacement = self.calc_displacement(assign=assign)
        else:
            displacement = self._trj.displacement

        displacement_time = self._trj.time - self._trj.time.iloc[0]

        derivs = OrderedDict(displacement=displacement, displacement_time=displacement_time)
        if assign:
            self._trj = self._trj.join(traja.TrajaDataFrame.from_records(derivs))
        return derivs

    def get_derivatives(self):
        """Get derivatives.

        Args:

        Returns:
          OrderedDict: Derivatives in dictionary.
          
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
        if not self._has_cols(['displacement', 'displacement_time']):
            derivs = self.calc_derivatives(assign=False)
            d = derivs['displacement']
            t = derivs['displacement_time']
        else:
            d = self._trj.displacement
            t = self._trj.displacement_time
            derivs = OrderedDict(displacement=d, displacement_time=t)
        v = d[1: len(d)] / t.diff()
        v.rename('speed')
        vt = t[1: len(t)].rename('speed_times')
        # Calculate linear acceleration
        a = v.diff() / vt.diff().rename('acceleration')
        at = vt[1: len(vt)].rename('accleration_times')
        data = OrderedDict(speed=v, speed_times=vt, acceleration=a, acceleration_times=at)
        derivs.update(data)
        return derivs

    @property
    def speed_intervals(self, faster_than=None, slower_than=None, interpolate_times=True):
        """Calculate speed time intervals.
        
        Returns a dictionary of time intervals where speed is slower and/or faster than specified values.

        Args:
          faster_than (float, optional): Minimum speed threshold. (Default value = None)
          slower_than (float or int, optional): Maximum speed threshold. (Default value = None)
          interpolate_times (bool, optional): Interpolate times between steps. (Default value = True)

        Returns:
          OrderedDict -- time intervals as dictionary.
          
          .. note::
          
            Implementation ported to Python, heavily inspired by Jim McLean's trajr package.

        """
        derivs = self.get_derivatives()

        if faster_than is not None:
            pass
        if slower_than is not None:
            pass

        # Calculate trajectory speeds
        speed = derivs.get('speed')
        times = derivs.get('speed_times')
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
            if len(stop_frames) > 0 and (len(start_frames) == 0 or stop_frames[0] < start_frames[0]):
                start_frames = np.append(1, start_frames)
            # Similarly, assume that interval can't extend past end of trajectory
            if len(stop_frames) == 0 or start_frames[len(start_frames) - 1] > stop_frames[len(stop_frames) - 1]:
                stop_frames = np.append(stop_frames, len(speed))

        stop_times = times[stop_frames]
        start_times = times[start_frames]

        if interpolate_times and len(start_frames) > 0:
            # TODO: Implement
            raise NotImplementedError()
            r = self.linear_interp_times(slower_than, faster_than, speed, times, start_frames, start_times)
            start_times = r[:, 0]
            stop_times = r[:, 1]

        durations = stop_times - start_times
        result = traja.TrajaDataFrame(OrderedDict(start_frame=start_frames,
                                           start_time=start_times,
                                           stop_frame=stop_frames,
                                           stop_time=stop_times,
                                           duration=durations))

        metadata = OrderedDict(slower_than=slower_than, faster_than=faster_than, derivs=derivs, trajectory=self._trj)
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
        df = self._trj[['x', 'y']].dropna()
        coords = df.values
        tracks_data = {'type': 'LineString',
                       'coordinates': coords}
        tracks_shape = shape(tracks_data)
        return tracks_shape

    def calc_displacement(self, assign=True):
        """Calculate displacement between consecutive indices.

        Args:
          assign(bool., optional): Assign displacement to TrajaDataFrame (Default value = True)

        Returns:
          pd.Series -- Displacement series.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.calc_displacement()
            0         NaN
            1    1.414214
            2    1.414214
            dtype: float64

        """
        displacement = np.sqrt(np.power(self._trj.x.shift() - self._trj.x, 2) +
                               np.power(self._trj.y.shift() - self._trj.y, 2))

        # dx = self._trj.x.diff()
        # dy = self._trj.y.diff()
        if assign:
            self._trj = self._trj.assign(displacement=displacement)
        return displacement

    def calc_angle(self, assign=True):
        """Calculate angle between steps as a function of displacement w.r.t x axis.

        Args:
          assign (bool, optional): Assign displacement to TrajaDataFrame (Default value = True)

        Returns:
          pd.Series -- Angle series.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.calc_angle()
            0     NaN
            1    45.0
            2    45.0
            dtype: float64

        """
        if not self._has_cols(['dx', 'displacement']):
            displacement = self.calc_displacement()
        else:
            displacement = self._trj.displacement

        angle = np.rad2deg(np.arccos(np.abs(self._trj.x.diff()) / displacement))
        if assign:
            self._trj['angle'] = angle
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
        self._trj[['x', 'y']] *= scale
        self._trj.__dict__['spatial_units'] = spatial_units


    def rediscretize(self, R):
        """Resample a trajectory to a constant step length. R is rediscretized step length.

        Args:
          R (float): Rediscretized step length (eg, 0.02)

        Returns:
          Rediscretized coordinates.
          
        .. note::
          
            Based on the appendix in Bovet and Benhamou, (1988) and @JimMcL's trajr implementation.
          
        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.rediscretize(1.)
                      x         y
            0  0.000000  1.000000
            1  0.707107  1.707107
            2  1.414214  2.414214

        """
        rt = self._rediscretize_points(R)

        if len(rt) < 2:
            raise RuntimeError(f"Step length {R} is too large for path (path length {len(self._trj)})")
        rt = traja.from_xy(rt)
        return rt

    def _rediscretize_points(self, R):
        """Helper function for `self.rediscretize`

        Args:
          R(float.): Rediscretized step length (eg, 0.02)

        Returns:
          Rediscretized coordinates.

        """
        # TODO: Implement with complex numbers
        points = self._trj[['x', 'y']].dropna().values.astype('float64')
        n_points = len(points)
        result = np.empty((128, 2))
        p0 = points[0]
        result[0] = p0
        I = 0
        j = 1

        while j <= n_points:
            # Find the first point k for which |p[k] - p_0| >= R
            k = np.NaN
            for i in range(j, n_points):  # range of search space for next point
                d = np.linalg.norm(points[i] - result[I])
                if d >= R:
                    k = i  # [j, n_points)
                    break
            if np.isnan(k):
                # End of path
                break

            # The next point may lie on the same segment
            j = k

            # The next point lies on the segment p[k-1], p[k]
            XI = result[I][0]
            xk_1 = points[k - 1, 0]
            YI = result[I][1]
            yk_1 = points[k - 1, 1]

            # a = 1 if points[k, 0] <= xk_1 else 0
            lambda_ = np.arctan2(points[k, 1] - yk_1, points[k, 0] - xk_1) # angle
            cos_l = np.cos(lambda_)
            sin_l = np.sin(lambda_)
            U = (XI - xk_1) * cos_l + (YI - yk_1) * sin_l
            V = (YI - yk_1) * cos_l - (XI - xk_1) * sin_l

            # Compute distance H between (X_{i+1}, Y_{i+1}) and (x_{k-1}, y_{k-1})
            H = U + np.sqrt(abs(R ** 2 - V ** 2))
            XIp1 = H * cos_l + xk_1
            YIp1 = H * sin_l + yk_1

            # Increase array size progressively to make the code run (significantly) faster
            if len(result) <= I + 1:
                result = np.concatenate((result, np.empty_like(result)))

            # Save the point
            result[I + 1] = np.array([XIp1, YIp1])
            I += 1

        # Truncate result
        result = result[:I + 1]
        return result

    def calc_heading(self, assign=True):
        """Calculate trajectory heading.

        Args:
          assign:  (Default value = True)

        Returns:

        ..doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.calc_heading()
            0     NaN
            1    45.0
            2    45.0
            Name: heading, dtype: float64

        """
        if not self._has_cols('angle'):
            angle = self.calc_angle(assign=True)
        else:
            angle = self._trj.angle
        df = self._trj
        dx = df.x.diff()
        dy = df.y.diff()
        # Get heading from angle
        mask = (dx > 0) & (dy >= 0)
        df.loc[mask, 'heading'] = angle[mask]
        mask = (dx >= 0) & (dy < 0)
        df.loc[mask, 'heading'] = -angle[mask]
        mask = (dx < 0) & (dy <= 0)
        df.loc[mask, 'heading'] = -(180 - angle[mask])
        mask = (dx <= 0) & (dy > 0)
        df.loc[mask, 'heading'] = (180 - angle[mask])
        if assign:
            self._trj['heading'] = df.heading
        return df.heading

    def calc_turn_angle(self, assign=True):
        """Calculate turn angle.


        Args:
          assign:  (Default value = True)

        Returns:

        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> df.traja.calc_turn_angle()
            0    NaN
            1    NaN
            2    0.0
            Name: turn_angle, dtype: float64

        """
        if 'heading' not in self._trj:
            heading = self.calc_heading(assign=False)
        else:
            heading = self._trj.heading
        turn_angle = heading.diff().rename('turn_angle')
        # Correction for 360-degree angle range
        turn_angle[turn_angle >= 180] -= 360
        turn_angle[turn_angle < -180] += 360
        if assign:
            self._trj['turn_angle'] = turn_angle
        return turn_angle