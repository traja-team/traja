from matplotlib.ticker import FormatStrFormatter

import traja
import matplotlib
import numpy as np

import matplotlib.pyplot as plt

from pandas.core.dtypes.common import (
    is_datetime_or_timedelta_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)
from traja.trajectory import coords_to_flow


def stylize_axes(ax):
    """Add top and right border to plot, set ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_tick_params(top="off", direction="out", width=1)
    ax.yaxis.set_tick_params(right="off", direction="out", width=1)


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

    plt.rc("font", family="serif")


def predict(xy, nb_steps=10, epochs=1000, batch_size=1, model="lstm"):
    """Method for training and visualizing LSTM with trajectory data."""
    if model is "lstm":
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

    GRAY = "#999999"
    self = accessor or {}
    if accessor:
        kwargs = self._get_plot_args(**kwargs)
    xlim = kwargs.pop("xlim", None)
    ylim = kwargs.pop("ylim", None)

    title = kwargs.pop("title", None)
    time_units = kwargs.pop("time_units", None)
    fps = kwargs.pop("fps", None)
    figsize = kwargs.pop("figsize", None)

    start, end = None, None
    coords = trj[["x", "y"]]
    time_col = traja.trajectory._get_time_col(trj)
    if time_col is "index":
        is_datetime = True
    else:
        is_datetime = (
            is_datetime_or_timedelta_dtype(trj[time_col]) if time_col else False
        )

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
    patch = patches.PathPatch(path, edgecolor=GRAY, facecolor="none", lw=3, alpha=0.3)
    ax.add_patch(patch)

    xs, ys = zip(*verts)

    if time_col is "index":
        # DatetimeIndex determines color
        colors = [ind for ind, x in enumerate(trj.index[:n_coords])]
    elif time_col and time_col is not "index":
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
            time_units = ""
    else:
        # Index/frame count is our only reference
        vmin = trj.index[0]
        vmax = trj.index[n_coords - 1]
        if not show_time:
            time_units = ""
    label = f"Time ({time_units})" if time_units else ""

    sc = ax.scatter(
        xs, ys, c=colors, s=25, cmap=plt.cm.viridis, alpha=0.7, vmin=vmin, vmax=vmax
    )

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim((coords.x.min(), coords.x.max()))
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim((coords.y.min(), coords.y.max()))

    if kwargs.pop("invert_yaxis", None):
        plt.gca().invert_yaxis()

    _label_axes(trj, ax)
    ax.set_title(title)
    ax.set_aspect("equal")

    # Number of color bar ticks
    # FIXME: Implement customizable
    CBAR_TICKS = 10 if n_coords > 20 else n_coords
    indices = np.linspace(0, n_coords - 1, CBAR_TICKS, endpoint=True, dtype=int)
    cbar = plt.colorbar(
        sc, fraction=0.046, pad=0.04, orientation="vertical", label=label
    )
    if time_col is "index":
        if is_datetime64_any_dtype(trj.index):
            cbar_labels = (
                trj.index[indices].strftime("%Y-%m-%d %H:%M:%S").values.astype(str)
            )
        elif is_timedelta64_dtype(trj.index):
            if time_units in ("s", "", None):
                cbar_labels = [round(x, 2) for x in trj.index[indices].total_seconds()]
            else:
                print("Time unit {} not yet implemented".format(time_units))
        else:
            raise NotImplementedError(
                "Indexing on {} is not yet implemented".format(type(trj.index))
            )
    elif time_col and is_timedelta64_dtype(trj[time_col]):
        cbar_labels = trj[time_col].iloc[indices].dt.total_seconds().values.astype(str)
    elif time_col and is_datetime:
        cbar_labels = (
            trj[time_col]
            .iloc[indices]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .values.astype(str)
        )
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


def _label_axes(trj, ax):
    if 'spatial_units' in trj.__dict__:
        ax.set_xlabel(trj.spatial_units)
        ax.set_ylabel(trj.spatial_units)
    return ax

def plot_quiver(trj, bins=None, quiverplot_kws = {}):
    """Plot average flow from each grid cell to neighbor.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        quiverplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.quiver`

    Returns:
        ax (:class:`~matplotlib.collections.PathCollection`): Axes of quiver plot
    """
    X, Y, U, V = coords_to_flow(trj, bins)
    Z = np.sqrt(U * U + V * V)

    fig, ax = plt.subplots()

    qp = ax.quiver(X, Y, U, V, units="width", **quiverplot_kws)
    ax = _label_axes(trj, ax)

    plt.show()
    return ax

def plot_contour(trj, bins=None, filled=True, quiver=True, contourplot_kws = {}, contourfplot_kws = {}, quiverplot_kws={}):
    """Plot average flow from each grid cell to neighbor.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        contourplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contour`
        contourfplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contourf`
        quiverplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.quiver`

    Returns:
        ax (:class:`~matplotlib.collections.PathCollection`): Axes of quiver plot
    """
    X, Y, U, V = coords_to_flow(trj, bins)
    Z = np.sqrt(U * U + V * V)

    fig, ax = plt.subplots()

    if filled:
        cfp = plt.contourf(X, Y, Z, **contourfplot_kws)
    cp = plt.contour(X, Y, Z, colors='k', linewidths=1, linestyles='solid', **contourplot_kws)
    if quiver:
        qp = ax.quiver(X, Y, U, V, units="width", **quiverplot_kws)

    ax = _label_axes(trj, ax)

    plt.show()
    return ax


def plot_surface(trj, bins=None, cmap='jet', **surfaceplot_kws):
    """Plot surface of flow from each grid cell to neighbor in 3D.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        cmap (str): color map
        surfaceplot_kws: Additional keyword arguments for :meth:`~mpl_toolkits.mplot3D.Axes3D.plot_surface`

    Returns:
        ax (:class:`~matplotlib.collections.PathCollection`): Axes of quiver plot
    """
    from mpl_toolkits.mplot3d import Axes3D
    X, Y, U, V = coords_to_flow(trj, bins)
    Z = np.sqrt(U * U + V * V)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=matplotlib.cm.coolwarm, linewidth=0, **surfaceplot_kws)

    ax = _label_axes(trj, ax)

    plt.show()
    return ax

def plot_stream(trj, bins=None, cmap='jet', contourfplot_kws = {}, contourplot_kws = {}, streamplot_kws={}):
    """Plot average flow from each grid cell to neighbor.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        contourplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contour`
        contourfplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contourf`
        streamplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.streamplot`

    Returns:
        ax (:class:`~matplotlib.collections.PathCollection`): Axes of quiver plot

    """
    X, Y, U, V = coords_to_flow(trj, bins)
    Z = np.sqrt(U * U + V * V)

    fig, ax = plt.subplots()

    cfp = plt.contourf(X, Y, Z, **contourfplot_kws)
    cp = plt.contour(X, Y, Z, colors='k', linewidths=1, linestyles='solid', **contourplot_kws)
    sp = ax.streamplot(X, Y, U, V, color=Z, cmap=cmap, **streamplot_kws)

    ax = _label_axes(trj, ax)

    plt.show()
    return ax

def plot_flow(trj,
              kind='quiver',
                 *args,
                 contourplot_kws = {},
                 contourfplot_kws={},
                 streamplot_kws ={},
                 quiverplot_kws={},
                 surfaceplot_kws={}):
    """Plot average flow from each grid cell to neighbor.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        kind (str): Choice of 'quiver','contourf','stream','surface'. Default is 'quiver'.
        contourplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contour`
        contourfplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contourf`
        streamplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.streamplot`
        quiverplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.quiver`
        surfaceplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.plot_surface`

    Returns:
        ax (:class:`~matplotlib.collections.PathCollection`): Axes of quiver plot
    """
    if kind is 'quiver':
        return plot_contour(trj, filled=False, *args, **quiverplot_kws)
    elif kind is 'contourf':
        return plot_contour(trj, *args, **quiverplot_kws)
    elif kind is 'stream':
        return plot_stream(trj,
                           *args,
                           contourplot_kws=contourplot_kws,
                           contourfplot_kws=contourfplot_kws,
                           streamplot_kws=streamplot_kws)
    elif kind is 'surface':
        return plot_surface(trj,
                            *args,
                            **surfaceplot_kws)
    else:
        raise NotImplementedError(f"Kind {kind} is not implemented.")


def trip_grid(
    trj,
    bins=32,
    log=False,
    spatial_units=None,
    normalize=False,
    hist_only=False,
    plot=False,
):
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
    bins = traja.trajectory._bins_to_tuple(trj, bins)
    # TODO: Add kde-based method for line-to-cell gridding
    df = trj[["x", "y"]].dropna()

    # Set aspect if `xlim` and `ylim` set.
    if (
        "xlim" in df.__dict__ and "ylim" in df.__dict__ and isinstance(df.xlim, tuple)
    ):  # TrajaDataFrame
        x0, x1 = df.xlim
        y0, y1 = df.ylim
    else:
        x0, x1 = (df.x.min(), df.x.max())
        y0, y1 = (df.y.min(), df.y.max())

    x_edges = np.linspace(x0, x1, num=bins[0])
    y_edges = np.linspace(y0, y1, num=bins[1])

    x, y = zip(*df.values)
    # FIXME: Remove redundant histogram calculation
    hist, x_edges, y_edges = np.histogram2d(
        x, y, bins=(x_edges, y_edges), normed=normalize
    )
    if log:
        hist = np.log(hist + np.e)
    if hist_only:  # TODO: Evaluate potential use cases or remove
        return hist, None
    fig, ax = plt.subplots()
    image = ax.imshow(hist, interpolation="bilinear", extent=[x0,x1,y0,y1])
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # TODO: Adjust colorbar ytick_labels to correspond with time
    label = 'Frames' if not log else '$ln(Frames)$'
    cbar = plt.colorbar(image, ax=ax, label='Frames')

    ax = _label_axes(trj, ax)

    plt.title("Time spent{}".format(" (Logarithmic)" if log else ""))

    if plot:
        plt.show()
    # TODO: Add method for most common locations in grid
    # peak_index = unravel_index(hist.argmax(), hist.shape)
    return hist, image
