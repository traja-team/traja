from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

import traja


@pd.api.extensions.register_dataframe_accessor("traja")
class TrajaAccessor(object):
    """Accessor for pandas DataFrame with trajectory-specific numerical and analytical functions.

    Access with `df.traja`."""

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._cache = {}  # Cache for expensive computations

    __axes = ["x", "y"]
    __optional_axes = ["z"]  # Optional third dimension

    @staticmethod
    def _set_axes(axes):
        if len(axes) not in (2, 3):
            raise ValueError(
                "TrajaAccessor requires 2 or 3 axes, got {}".format(len(axes))
            )
        TrajaAccessor.__axes = axes[:2]  # First two are required
        if len(axes) == 3:
            TrajaAccessor.__optional_axes = [axes[2]]

    def _strip(self, text):
        try:
            return text.strip()
        except AttributeError:
            return pd.to_numeric(text, errors="coerce")

    @staticmethod
    def _validate(obj):
        if (
            TrajaAccessor.__axes[0] not in obj.columns
            or TrajaAccessor.__axes[1] not in obj.columns
        ):
            raise AttributeError(
                "Must have '{}' and '{}'.".format(*TrajaAccessor.__axes)
            )

    def _has_z(self) -> bool:
        """Check if trajectory has z coordinate."""
        return "z" in self._obj.columns

    @property
    def center(self):
        """Return the center point of this trajectory.

        Returns:
            tuple: (x, y) for 2D or (x, y, z) for 3D trajectories
        """
        x = self._obj.x
        y = self._obj.y
        if self._has_z():
            z = self._obj.z
            return float(x.mean()), float(y.mean()), float(z.mean())
        return float(x.mean()), float(y.mean())

    @property
    def bounds(self):
        """Return limits of dimensions.

        Returns:
            tuple: ((xmin, xmax), (ymin, ymax)) for 2D or ((xmin, xmax), (ymin, ymax), (zmin, zmax)) for 3D
        """
        xlim = self._obj.x.min(), self._obj.x.max()
        ylim = self._obj.y.min(), self._obj.y.max()
        if self._has_z():
            zlim = self._obj.z.min(), self._obj.z.max()
            return (xlim, ylim, zlim)
        return (xlim, ylim)

    def night(self, begin: str = "19:00", end: str = "7:00"):
        """Get nighttime dataset between `begin` and `end`.

        Args:
          begin (str):  (Default value = '19:00')
          end (str):  (Default value = '7:00')

        Returns:
          trj (:class:`~traja.frame.TrajaDataFrame`): Trajectory during night.

        """
        return self.between(begin, end)

    def day(self, begin: str = "7:00", end: str = "19:00"):
        """Get daytime dataset between `begin` and `end`.

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
        if time_col == "index":
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

    def rediscretize_points(self, R, **kwargs):
        """Rediscretize points"""
        return traja.trajectory.rediscretize_points(self._obj, R=R, **kwargs)

    @property
    def speed(self):
        """Calculate instantaneous speed.

        Returns:
            pd.Series: Speed at each time point

        .. note::
            Requires time column or fps metadata
        """
        derivs = traja.trajectory.get_derivatives(self._obj)
        return derivs['speed']

    @property
    def displacement(self):
        """Calculate displacement between consecutive points.

        Returns:
            pd.Series: Displacement at each step (supports 2D and 3D)
        """
        return traja.trajectory.calc_displacement(self._obj)

    def summary(self):
        """Generate summary statistics for the trajectory.

        Returns:
            dict: Dictionary containing trajectory statistics

        .. doctest::

            >>> df = traja.TrajaDataFrame({'x':[0,1,2],'y':[1,2,3]})
            >>> stats = df.traja.summary()
            >>> stats['n_points']
            3
            >>> stats['distance'] > 0
            True

        """
        stats = {}
        stats['n_points'] = len(self._obj)
        stats['center'] = self.center
        stats['bounds'] = self.bounds
        stats['distance'] = traja.trajectory.distance(self._obj)
        stats['length'] = traja.trajectory.length(self._obj)
        stats['dimensionality'] = '3D' if self._has_z() else '2D'

        if self._get_time_col():
            try:
                derivs = traja.trajectory.get_derivatives(self._obj)
                stats['mean_speed'] = float(derivs['speed'].mean())
                stats['max_speed'] = float(derivs['speed'].max())
                stats['mean_acceleration'] = float(derivs['acceleration'].mean())
            except:
                pass  # Time data not suitable for derivatives

        return stats

    def trip_grid(
        self,
        bins: Union[int, tuple] = 10,
        log: bool = False,
        spatial_units=None,
        normalize: bool = False,
        hist_only: bool = False,
        plot: bool = True,
        **kwargs,
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
            spatial_units=self._obj.get("spatial_units", "m"),
            normalize=normalize,
            hist_only=hist_only,
            plot=plot,
            **kwargs,
        )
        return hist, image

    def plot(self, n_coords: int = None, show_time=False, **kwargs):
        """Plot trajectory over period.

        Args:
          n_coords (int): Number of coordinates to plot
          **kwargs: additional keyword arguments to :meth:`matplotlib.axes.Axes.scatter`

        Returns:
            ax (:class:`~matplotlib.axes.Axes`): Axes of plot
        """
        ax = traja.plotting.plot(
            trj=self._obj,
            accessor=self,
            n_coords=n_coords,
            show_time=show_time,
            **kwargs,
        )
        return ax

    def plot_3d(self, **kwargs):
        """Plot 3D trajectory for single identity over period.

        Args:
        trj (:class:`traja.TrajaDataFrame`): trajectory
        n_coords (int, optional): Number of coordinates to plot
        **kwargs: additional keyword arguments to :meth:`matplotlib.axes.Axes.scatter`

        Returns:
            collection (:class:`~matplotlib.collections.PathCollection`): collection that was plotted

        .. note::
            Takes a while to plot large trajectories. Consider using first::

                rt = trj.traja.rediscretize(R=1.) # Replace R with appropriate step length
                rt.traja.plot_3d()

        """
        ax = traja.plotting.plot_3d(trj=self._obj, **kwargs)
        return ax

    def plot_flow(self, kind="quiver", **kwargs):
        """Plot grid cell flow.

        Args:
          kind (str): Kind of plot (eg, 'quiver','surface','contour','contourf','stream')
          **kwargs: additional keyword arguments to :meth:`matplotlib.axes.Axes.scatter`

        Returns:
            ax (:class:`~matplotlib.axes.Axes`): Axes of plot

        """
        ax = traja.plotting.plot_flow(trj=self._obj, kind=kind, **kwargs)
        return ax

    def plot_collection(self, colors=None, **kwargs):
        return traja.plotting.plot_collection(
            self._obj, id_col=self._id_col, colors=colors, **kwargs
        )

    def apply_all(self, method, id_col=None, **kwargs):
        """Applies method to all trajectories and returns grouped dataframes or series"""
        id_col = id_col or getattr(self, "_id_col", "id")
        return self._obj.groupby(by=id_col).apply(method, **kwargs)

    def _has_cols(self, cols: list):
        return traja.trajectory._has_cols(self._obj, cols)

    @property
    def xyz(self):
        """Returns a :class:`numpy.ndarray` of x,y,z coordinates (if z exists).

        Returns:
          xyz (:class:`numpy.ndarray`) -- x,y,z coordinates or x,y if z doesn't exist

        """
        if self._has_z():
            if self._has_cols(["x", "y", "z"]):
                xyz = self._obj[["x", "y", "z"]].values
                return xyz
            else:
                raise KeyError("'x', 'y', and 'z' are not in the dataframe.")
        else:
            return self.xy

    @property
    def xy(self):
        """Returns a :class:`numpy.ndarray` of x,y coordinates.

        Args:
            split (bool): Split into separate x and y :class:`numpy.ndarrays`

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
            raise KeyError("'x' and 'y' are not in the dataframe.")

    def _check_has_time(self):
        """Check for presence of displacement time column."""
        time_col = self._get_time_col()
        if time_col is None:
            raise ValueError("Missing time information in trajectory.")

    def __getattr__(self, name):
        """Catch all method calls which are not defined and forward to modules."""

        def method(*args, **kwargs):
            if name in traja.plotting.__all__:
                return getattr(traja.plotting, name)(self._obj, *args, **kwargs)
            elif name in traja.trajectory.__all__:
                return getattr(traja.plotting, name)(self._obj, *args, **kwargs)
            elif name in dir(self):
                return getattr(self, name)(*args)(**kwargs)
            else:
                raise AttributeError(f"{name} attribute not defined")

        return method

    def transitions(self, *args, **kwargs):
        """Calculate transition matrix"""
        return traja.transitions(self._obj, *args, **kwargs)

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

    def speed_intervals(
        self,
        faster_than: Union[float, int] = None,
        slower_than: Union[float, int] = None,
    ):
        """Returns ``TrajaDataFrame`` with speed time intervals.

        Returns a dataframe of time intervals where speed is slower and/or faster than specified values.

        Args:
          faster_than (float, optional): Minimum speed threshold. (Default value = None)
          slower_than (float or int, optional): Maximum speed threshold. (Default value = None)

        Returns:
          result (:class:`~pandas.DataFrame`) -- time intervals as dataframe

        .. note::

            Implementation ported to Python, heavily inspired by Jim McLean's trajr package.

        """
        result = traja.trajectory.speed_intervals(self._obj, faster_than, slower_than)
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

    def to_csv(self, filepath: str, **kwargs):
        """Export trajectory to CSV file.

        Args:
            filepath: Path to save CSV file
            **kwargs: Additional arguments passed to pandas.to_csv()

        Returns:
            None
        """
        self._obj.to_csv(filepath, **kwargs)

    def to_hdf(self, filepath: str, key: str = 'trajectory', **kwargs):
        """Export trajectory to HDF5 file.

        Args:
            filepath: Path to save HDF5 file
            key: HDF5 group identifier (default: 'trajectory')
            **kwargs: Additional arguments passed to pandas.to_hdf()

        Returns:
            None
        """
        self._obj.to_hdf(filepath, key=key, **kwargs)

    def to_npy(self, filepath: str, columns: Optional[list] = None):
        """Export trajectory to NumPy .npy file.

        Args:
            filepath: Path to save .npy file
            columns: List of columns to export (default: ['x', 'y'] or ['x', 'y', 'z'])

        Returns:
            None
        """
        if columns is None:
            if self._has_z():
                columns = ['x', 'y', 'z']
            else:
                columns = ['x', 'y']

        data = self._obj[columns].values
        np.save(filepath, data)

    def to_tensor(self, columns: Optional[list] = None):
        """Convert trajectory to PyTorch tensor (if torch is available).

        Args:
            columns: List of columns to include (default: ['x', 'y'] or ['x', 'y', 'z'])

        Returns:
            torch.Tensor: Trajectory as tensor, or numpy array if torch not available

        .. note::
            Requires PyTorch. Install with: pip install torch

        """
        if columns is None:
            if self._has_z():
                columns = ['x', 'y', 'z']
            else:
                columns = ['x', 'y']

        data = self._obj[columns].values

        try:
            import torch
            return torch.from_numpy(data).float()
        except ImportError:
            import warnings
            warnings.warn("PyTorch not installed. Returning numpy array instead.")
            return data

    def extract_features(self) -> pd.DataFrame:
        """Extract common features for machine learning.

        Returns:
            pd.DataFrame: Feature matrix with columns like displacement, speed,
                         turn_angle, acceleration, etc.

        .. note::
            Useful for feeding into classical ML models or as additional features for DL.

        """
        features = pd.DataFrame(index=self._obj.index)

        # Spatial features
        features['displacement'] = self.displacement
        if self._has_z():
            features['displacement_xy'] = np.sqrt(
                np.power(self._obj.x.diff(), 2) + np.power(self._obj.y.diff(), 2)
            )
            features['displacement_z'] = self._obj.z.diff()

        # Angular features (2D only)
        if not self._has_z():
            features['turn_angle'] = traja.trajectory.calc_turn_angle(self._obj)
            features['heading'] = traja.trajectory.calc_heading(self._obj)

        # Temporal features (if time available)
        if self._get_time_col():
            try:
                derivs = traja.trajectory.get_derivatives(self._obj)
                features['speed'] = derivs['speed']
                features['acceleration'] = derivs['acceleration']
            except:
                pass

        return features.fillna(0)

    def augment_rotate(self, angle: float = None) -> "traja.TrajaDataFrame":
        """Rotate trajectory by angle (in degrees) for data augmentation.

        Args:
            angle (float, optional): Rotation angle in degrees. If None, random angle [0, 360).

        Returns:
            traja.TrajaDataFrame: Rotated trajectory

        .. note::
            Useful for training rotation-invariant deep learning models.

        """
        import numpy as np
        from traja import TrajaDataFrame

        if angle is None:
            angle = np.random.uniform(0, 360)

        theta = np.radians(angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Rotation matrix
        rotated = self._obj.copy()
        x = self._obj.x.values
        y = self._obj.y.values

        rotated['x'] = x * cos_theta - y * sin_theta
        rotated['y'] = x * sin_theta + y * cos_theta

        # Z coordinate is not rotated (vertical axis)

        return TrajaDataFrame(rotated)

    def augment_noise(self, sigma: float = 0.1) -> "traja.TrajaDataFrame":
        """Add Gaussian noise to trajectory coordinates for data augmentation.

        Args:
            sigma (float): Standard deviation of Gaussian noise relative to coordinate scale.
                          Default 0.1 means 10% of the coordinate range.

        Returns:
            traja.TrajaDataFrame: Noisy trajectory

        .. note::
            Useful for making deep learning models robust to measurement noise.

        """
        import numpy as np
        from traja import TrajaDataFrame

        noisy = self._obj.copy()

        # Add noise proportional to the coordinate ranges
        x_range = self._obj.x.max() - self._obj.x.min()
        y_range = self._obj.y.max() - self._obj.y.min()

        noisy['x'] = self._obj.x + np.random.normal(0, sigma * x_range, len(self._obj))
        noisy['y'] = self._obj.y + np.random.normal(0, sigma * y_range, len(self._obj))

        if self._has_z():
            z_range = self._obj.z.max() - self._obj.z.min()
            noisy['z'] = self._obj.z + np.random.normal(0, sigma * z_range, len(self._obj))

        return TrajaDataFrame(noisy)

    def augment_reverse(self) -> "traja.TrajaDataFrame":
        """Reverse trajectory temporally for data augmentation.

        Returns:
            traja.TrajaDataFrame: Time-reversed trajectory

        .. note::
            Useful for data augmentation when temporal direction is not important.

        """
        from traja import TrajaDataFrame

        reversed_df = self._obj.iloc[::-1].reset_index(drop=True)
        return TrajaDataFrame(reversed_df)

    def augment_scale(self, factor: float = None) -> "traja.TrajaDataFrame":
        """Scale trajectory coordinates for data augmentation.

        Args:
            factor (float, optional): Scaling factor. If None, random factor in [0.8, 1.2].

        Returns:
            traja.TrajaDataFrame: Scaled trajectory

        .. note::
            Useful for scale-invariant deep learning models.

        """
        import numpy as np
        from traja import TrajaDataFrame

        if factor is None:
            factor = np.random.uniform(0.8, 1.2)

        scaled = self._obj.copy()
        scaled['x'] = self._obj.x * factor
        scaled['y'] = self._obj.y * factor

        if self._has_z():
            scaled['z'] = self._obj.z * factor

        return TrajaDataFrame(scaled)

    def augment_subsample(self, step: int = None) -> "traja.TrajaDataFrame":
        """Subsample trajectory by taking every nth point for data augmentation.

        Args:
            step (int, optional): Subsample step. If None, random step in [2, 5].

        Returns:
            traja.TrajaDataFrame: Subsampled trajectory

        .. note::
            Useful for training models on different temporal resolutions.

        """
        import numpy as np
        from traja import TrajaDataFrame

        if step is None:
            step = np.random.randint(2, 6)

        subsampled = self._obj.iloc[::step].reset_index(drop=True)
        return TrajaDataFrame(subsampled)

    def pad_trajectory(
        self, target_length: int, mode: str = 'edge', **kwargs
    ) -> "traja.TrajaDataFrame":
        """Pad trajectory to target length for deep learning batching.

        Args:
            target_length (int): Desired trajectory length
            mode (str): Padding mode - 'edge' (repeat last value), 'constant' (zeros),
                       'linear' (linear extrapolation). Default 'edge'.
            **kwargs: Additional arguments passed to padding functions

        Returns:
            traja.TrajaDataFrame: Padded trajectory

        Raises:
            ValueError: If target_length is less than current length

        .. note::
            Essential for batching variable-length trajectories in deep learning.

        """
        import numpy as np
        from traja import TrajaDataFrame

        current_length = len(self._obj)

        if target_length < current_length:
            raise ValueError(
                f"target_length ({target_length}) must be >= current length ({current_length}). "
                "Use truncate_trajectory() to shorten."
            )

        if target_length == current_length:
            return TrajaDataFrame(self._obj.copy())

        n_pad = target_length - current_length
        padded = self._obj.copy()

        if mode == 'edge':
            # Repeat last row
            last_row = self._obj.iloc[[-1]]
            padding = pd.concat([last_row] * n_pad, ignore_index=True)

        elif mode == 'constant':
            # Pad with zeros
            padding = pd.DataFrame(
                np.zeros((n_pad, len(self._obj.columns))),
                columns=self._obj.columns
            )

        elif mode == 'linear':
            # Linear extrapolation from last two points
            if current_length < 2:
                raise ValueError("Linear extrapolation requires at least 2 points")

            last_two = self._obj.iloc[-2:].copy()
            delta = last_two.iloc[1] - last_two.iloc[0]

            padding_data = []
            for i in range(1, n_pad + 1):
                new_row = last_two.iloc[1] + delta * i
                padding_data.append(new_row)

            padding = pd.DataFrame(padding_data)

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'edge', 'constant', or 'linear'.")

        # Concatenate original and padding
        result = pd.concat([padded, padding], ignore_index=True)
        return TrajaDataFrame(result)

    def truncate_trajectory(
        self, target_length: int, mode: str = 'end'
    ) -> "traja.TrajaDataFrame":
        """Truncate trajectory to target length for deep learning batching.

        Args:
            target_length (int): Desired trajectory length
            mode (str): Truncation mode - 'end' (keep first N), 'start' (keep last N),
                       'random' (random starting point). Default 'end'.

        Returns:
            traja.TrajaDataFrame: Truncated trajectory

        Raises:
            ValueError: If target_length is greater than current length

        .. note::
            Essential for batching variable-length trajectories in deep learning.

        """
        import numpy as np
        from traja import TrajaDataFrame

        current_length = len(self._obj)

        if target_length > current_length:
            raise ValueError(
                f"target_length ({target_length}) must be <= current length ({current_length}). "
                "Use pad_trajectory() to extend."
            )

        if target_length == current_length:
            return TrajaDataFrame(self._obj.copy())

        if mode == 'end':
            # Keep first target_length points
            truncated = self._obj.iloc[:target_length]

        elif mode == 'start':
            # Keep last target_length points
            truncated = self._obj.iloc[-target_length:]

        elif mode == 'random':
            # Random starting point
            max_start = current_length - target_length
            start_idx = np.random.randint(0, max_start + 1)
            truncated = self._obj.iloc[start_idx:start_idx + target_length]

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'end', 'start', or 'random'.")

        return TrajaDataFrame(truncated.reset_index(drop=True))

    def normalize_trajectory(
        self, scale: bool = True, center: bool = True
    ) -> "traja.TrajaDataFrame":
        """Normalize trajectory coordinates for deep learning.

        Args:
            scale (bool): Scale to unit variance. Default True.
            center (bool): Center to zero mean. Default True.

        Returns:
            traja.TrajaDataFrame: Normalized trajectory

        .. note::
            Normalization improves deep learning convergence and performance.

        """
        import numpy as np
        from traja import TrajaDataFrame

        normalized = self._obj.copy()

        coords = ['x', 'y']
        if self._has_z():
            coords.append('z')

        for coord in coords:
            values = self._obj[coord].values

            if center:
                values = values - values.mean()

            if scale:
                std = values.std()
                if std > 0:
                    values = values / std

            normalized[coord] = values

        return TrajaDataFrame(normalized)

    def plot_interactive(self, **kwargs):
        """Create interactive 2D or 3D trajectory plot using plotly.

        Args:
            **kwargs: Additional arguments passed to plotly

        Returns:
            plotly.graph_objs.Figure: Interactive plot figure

        .. note::
            Requires plotly: pip install plotly

        Example:
            >>> fig = df.traja.plot_interactive()
            >>> fig.show()

        """
        try:
            import plotly.graph_objs as go
        except ImportError:
            raise ImportError(
                "plotly is required for interactive plotting. "
                "Install with: pip install plotly"
            )

        if self._has_z():
            # 3D interactive plot
            fig = go.Figure(data=[go.Scatter3d(
                x=self._obj.x,
                y=self._obj.y,
                z=self._obj.z,
                mode='lines+markers',
                marker=dict(size=3, color=np.arange(len(self._obj)), colorscale='Viridis'),
                line=dict(color='rgba(100, 100, 100, 0.5)', width=2),
                **kwargs
            )])
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                title='3D Trajectory'
            )
        else:
            # 2D interactive plot
            fig = go.Figure(data=[go.Scatter(
                x=self._obj.x,
                y=self._obj.y,
                mode='lines+markers',
                marker=dict(size=5, color=np.arange(len(self._obj)), colorscale='Viridis'),
                line=dict(color='rgba(100, 100, 100, 0.5)', width=2),
                **kwargs
            )])
            fig.update_layout(
                xaxis_title='X',
                yaxis_title='Y',
                title='Trajectory',
                hovermode='closest'
            )

        return fig

    def plot_heatmap(self, bins: int = 50, cmap: str = 'hot', **kwargs):
        """Plot 2D heatmap showing time spent in each location.

        Args:
            bins (int): Number of bins for 2D histogram. Default 50.
            cmap (str): Matplotlib colormap name. Default 'hot'.
            **kwargs: Additional arguments passed to plt.imshow

        Returns:
            matplotlib.axes.Axes: Heatmap axes

        Example:
            >>> ax = df.traja.plot_heatmap(bins=30)
            >>> plt.show()

        """
        import matplotlib.pyplot as plt

        # Create 2D histogram
        heatmap, xedges, yedges = np.histogram2d(
            self._obj.x, self._obj.y, bins=bins
        )

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            heatmap.T,
            extent=extent,
            origin='lower',
            cmap=cmap,
            aspect='auto',
            **kwargs
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Trajectory Heatmap')
        plt.colorbar(im, ax=ax, label='Time spent')

        return ax

    def plot_speed(self, **kwargs):
        """Plot speed over time.

        Args:
            **kwargs: Additional arguments passed to plt.plot

        Returns:
            matplotlib.axes.Axes: Speed plot axes

        Example:
            >>> ax = df.traja.plot_speed()
            >>> plt.show()

        """
        import matplotlib.pyplot as plt

        speed = self.speed

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(speed, **kwargs)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Speed')
        ax.set_title('Speed over Time')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_acceleration(self, **kwargs):
        """Plot acceleration over time.

        Args:
            **kwargs: Additional arguments passed to plt.plot

        Returns:
            matplotlib.axes.Axes: Acceleration plot axes

        Raises:
            ValueError: If time column is not available

        Example:
            >>> ax = df.traja.plot_acceleration()
            >>> plt.show()

        """
        import matplotlib.pyplot as plt

        time_col = self._get_time_col()
        if not time_col:
            raise ValueError("Time column required for acceleration calculation")

        derivs = traja.trajectory.get_derivatives(self._obj)

        if 'acceleration' not in derivs.columns:
            raise ValueError("Could not calculate acceleration")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(derivs['acceleration'], **kwargs)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Acceleration')
        ax.set_title('Acceleration over Time')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_trajectory_components(self, figsize=(12, 8)):
        """Plot comprehensive trajectory analysis with multiple subplots.

        Shows: trajectory path, x/y components, speed, and displacement.

        Args:
            figsize (tuple): Figure size. Default (12, 8).

        Returns:
            matplotlib.figure.Figure: Figure with subplots

        Example:
            >>> fig = df.traja.plot_trajectory_components()
            >>> plt.show()

        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Trajectory path
        axes[0, 0].plot(self._obj.x, self._obj.y, '-', alpha=0.6)
        axes[0, 0].plot(self._obj.x.iloc[0], self._obj.y.iloc[0], 'go', markersize=10, label='Start')
        axes[0, 0].plot(self._obj.x.iloc[-1], self._obj.y.iloc[-1], 'ro', markersize=10, label='End')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_title('Trajectory Path')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: X and Y over time
        axes[0, 1].plot(self._obj.x, label='X', alpha=0.7)
        axes[0, 1].plot(self._obj.y, label='Y', alpha=0.7)
        axes[0, 1].set_xlabel('Time step')
        axes[0, 1].set_ylabel('Position')
        axes[0, 1].set_title('X and Y Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Speed
        speed = self.speed
        axes[1, 0].plot(speed, color='orange')
        axes[1, 0].set_xlabel('Time step')
        axes[1, 0].set_ylabel('Speed')
        axes[1, 0].set_title('Speed over Time')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Displacement
        displacement = self.displacement
        axes[1, 1].plot(displacement, color='green')
        axes[1, 1].set_xlabel('Time step')
        axes[1, 1].set_ylabel('Displacement')
        axes[1, 1].set_title('Displacement over Time')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

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
        """Returns ``Series`` with angle between steps as a function of displacement with regard to x axis.

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
        if not isinstance(R, (int, float)):
            raise ValueError(f"R must be provided as float or int")
        rt = traja.trajectory.rediscretize_points(self._obj, R)
        self._transfer_metavars(rt)
        return rt

    def grid_coordinates(self, **kwargs):
        return traja.grid_coordinates(self._obj, **kwargs)

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
