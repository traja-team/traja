import traja
import numpy as np

from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype, is_datetime64_any_dtype, is_timedelta64_dtype

def stylize_axes(ax):
    """Add top and right border to plot, set ticks."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)


def shift_xtick_labels(xtick_labels, first_index=None):
    for idx, x in enumerate(xtick_labels):
        label = x.get_text()
        xtick_labels[idx].set_text(str(int(label) + 1))
        if first_index is not None:
            xtick_labels[0] = first_index
    return xtick_labels


def sans_serif():
    """Convenience function for changing plot text to serif font."""
    import matplotlib.pyplot as plt
    plt.rc('font', family='serif')


def predict(xy, nb_steps=10, epochs=1000, batch_size=1, model='lstm'):
    """Method for training and visualizing LSTM with trajectory data."""
    if model is 'lstm':
        from traja.models.nn import TrajectoryLSTM
        TrajectoryLSTM(xy, nb_steps=nb_steps, epochs=epochs, batch_size=batch_size)

def plot(trj, n_coords: int = None, show_time=False, accessor=None, **kwargs):
    """Plot trajectory for single animal over period.

    Args:
      trj (:class:`traja.TrajaDataFrame`): trajectory
      n_coords (int): Number of coordinates to plot
      show_time (bool): Show colormap as time
      accessor (:class:`~traja.accessor.TrajaAccessor`, optional): TrajaAccessor instance
      **kwargs: additional keyword arguments to :meth:`matplotlib.axes.Axes.scatter`

    Returns:
        ax (:class:`~matplotlib.collections.PathCollection`): Axes of plot

    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.path import Path

    GRAY = '#999999'
    self = accessor or {}
    if accessor:
        kwargs = self._get_plot_args(**kwargs)
    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)
    spatial_units = trj.__dict__.get('spatial_units', None)
    xlabel = kwargs.pop('xlabel', None) or f'x ({spatial_units})' if spatial_units else ''
    ylabel = kwargs.pop('ylabel', None) or f'y ({spatial_units})' if spatial_units else ''
    title = kwargs.pop('title', None)
    time_units = kwargs.pop('time_units', None)
    fps = kwargs.pop('fps', None)
    figsize = kwargs.pop('figsize', None)

    start, end = None, None
    coords = trj[['x', 'y']]
    time_col = traja.trajectory._get_time_col(trj)
    if time_col is 'index':
        is_datetime = True
    else:
        is_datetime = is_datetime_or_timedelta_dtype(trj[time_col]) if time_col else False

    if n_coords is None:
        # Plot all coords
        start, end = 0, len(coords)
        verts = coords.iloc[:end].values
    else:
        # Plot first `n_coords`
        start, end = 0, n_coords
        verts = coords.iloc[:n_coords].values

    n_coords = len(verts)

    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    path = Path(verts, codes)

    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.draw()
    patch = patches.PathPatch(path, edgecolor=GRAY, facecolor='none', lw=3, alpha=0.3)
    ax.add_patch(patch)

    xs, ys = zip(*verts)

    if time_col is 'index':
        # DatetimeIndex determines color
        colors = [ind for ind, x in enumerate(trj.index[:n_coords])]
    elif time_col and time_col is not 'index':
        # `time_col` determines color
        colors = [ind for ind, x in enumerate(trj[time_col].iloc[:n_coords])]
    else:
        # Frame count determines color
        colors = trj.index[:n_coords]

    if time_col:
        # TODO: Calculate fps if not in datetime
        vmin = min(colors)
        vmax = max(colors)
        if is_datetime:
            # Show timestamps without units
            time_units = ''
    else:
        # Index/frame count is our only reference
        vmin = trj.index[0]
        vmax = trj.index[n_coords - 1]
        if not show_time:
            time_units = ''
    label = f"Time ({time_units})" if time_units else ""

    sc = ax.scatter(xs, ys, c=colors, s=25, cmap=plt.cm.viridis, alpha=0.7, vmin=vmin, vmax=vmax)

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

    # Number of color bar ticks
    # FIXME: Implement customizable
    CBAR_TICKS = 10 if n_coords > 20 else n_coords
    indices = np.linspace(0, n_coords - 1, CBAR_TICKS, endpoint=True, dtype=int)
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04, orientation='vertical', label=label)
    if time_col is 'index':
        if is_datetime64_any_dtype(trj.index):
            cbar_labels = trj.index[indices].strftime("%Y-%m-%d %H:%M:%S").values.astype(str)
        elif is_timedelta64_dtype(trj.index):
            if time_units in ('s', '', None):
                cbar_labels = [round(x,2) for x in trj.index[indices].total_seconds()]
            else:
                print("Time unit {} not yet implemented".format(time_units))
        else:
            raise NotImplementedError("Indexing on {} is not yet implemented".format(type(trj.index)))
    elif time_col and is_timedelta64_dtype(trj[time_col]):
        cbar_labels = trj[time_col].iloc[indices].dt.total_seconds().values.astype(str)
    elif time_col and is_datetime:
        cbar_labels = trj[time_col].iloc[indices].dt.strftime("%Y-%m-%d %H:%M:%S").values.astype(str)
    else:
        # Convert frames to time
        cbar_labels = trj.index[indices].values
        if fps is not None and fps > 0 and fps is not 1 and show_time:
            cbar_labels = cbar_labels / fps
    cbar.set_ticks(indices)
    cbar.set_ticklabels(cbar_labels)
    plt.tight_layout()
    plt.show()
    return ax


def trip_grid(trj, bins=16, log=False, spatial_units=None, normalize=False, hist_only=False, plot=False):
    """Generate a heatmap of time spent by point-to-cell gridding.

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
    import matplotlib.pyplot as plt
    # TODO: Add kde-based method for line-to-cell gridding
    df = trj[['x', 'y']].dropna()

    # Set aspect if `xlim` and `ylim` set.
    if 'xlim' in df.__dict__ and 'ylim' in df.__dict__ and isinstance(df.xlim,tuple): # TrajaDataFrame
        x0, x1 = df.xlim
        y0, y1 = df.ylim
    else:
        x0, x1 = (df.x.min(), df.x.max())
        y0, y1 = (df.y.min(), df.y.max())
    aspect = (y1 - y0) / (x1 - x0)
    x_edges = np.linspace(x0, x1, num=bins)
    y_edges = np.linspace(y0, y1, num=int(bins / aspect))

    x, y = zip(*df.values)
    # # TODO: Remove redundant histogram calculation
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=(x_edges, y_edges), density=normalize)
    if log:
        hist = np.log(hist + np.e)
    if hist_only:
        return hist, None
    fig, ax = plt.subplots()
    image = plt.imshow(hist, interpolation='bilinear')
    # TODO: Adjust colorbar ytick_labels to correspond with time
    cbar = plt.colorbar(image, ax=ax)
    ax.set_xlabel("x{}".format(" (" + spatial_units + ")" if spatial_units else ""))
    ax.set_ylabel("y{}".format(" (" + spatial_units + ")" if spatial_units else ""))
    plt.title("Time spent{}".format(' (Logarithmic)' if log else ''))
    plt.tight_layout()
    if plot:
        plt.show()
    # TODO: Add method for most common locations in grid
    # peak_index = unravel_index(hist.argmax(), hist.shape)
    return hist, image

