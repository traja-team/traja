from typing import Union, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_datetime_or_timedelta_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)

import traja
from traja import TrajaDataFrame
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


def predict(
    xy: np.ndarray,
    nb_steps: int = 10,
    epochs: int = 1000,
    batch_size: int = 1,
    model="lstm",
):
    """Method for training and visualizing LSTM with trajectory data."""
    if model is "lstm":
        from traja.models.nn import TrajectoryLSTM

        TrajectoryLSTM(xy, nb_steps=nb_steps, epochs=epochs, batch_size=batch_size)


def bar_plot(trj: TrajaDataFrame, bins: Union[int, tuple] = None, **kwargs):
    """Plot trajectory for single animal over period.

    Args:
      trj (:class:`traja.TrajaDataFrame`): trajectory
      bins
      **kwargs: additional keyword arguments to :meth:`mpl_toolkits.mplot3d.Axed3D.bar3d`

    Returns:
        ax (:class:`~matplotlib.collections.PathCollection`): Axes of plot

    """
    # TODO: Add time component
    from mpl_toolkits.mplot3d import Axes3D

    bins = traja.trajectory._bins_to_tuple(trj, bins)

    X, Y, U, V = coords_to_flow(trj, bins)
    Z = np.sqrt(U * U + V * V)

    hist, _ = trip_grid(trj, bins, hist_only=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_aspect("equal")
    X = X.flatten("F")
    Y = Y.flatten("F")
    ax.bar3d(
        X,
        Y,
        np.zeros_like(X),
        1,
        1,
        hist.flatten(),
        zsort="average",
        shade=True,
        **kwargs,
    )
    ax.set(xlabel="x", ylabel="y", zlabel="Frames")

    plt.show()
    return ax


def plot(
    trj: TrajaDataFrame,
    n_coords: Optional[int] = None,
    show_time: bool = False,
    accessor: Optional[traja.TrajaAccessor] = None,
    **kwargs,
):
    """Plot trajectory for single animal over period.

    Args:
      trj (:class:`traja.TrajaDataFrame`): trajectory
      n_coords (int, optional): Number of coordinates to plot
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
    if not xlim or not ylim:
        xlim, ylim = traja.trajectory._get_xylim(trj)

    title = kwargs.pop("title", None)
    time_units = kwargs.pop("time_units", "s")
    fps = kwargs.pop("fps", None)
    figsize = kwargs.pop("figsize", None)

    start, end = None, None
    coords = trj[["x", "y"]]
    time_col = traja.trajectory._get_time_col(trj)

    if time_col is "index":
        is_datetime = True
    else:
        is_datetime = is_datetime64_any_dtype(trj[time_col]) if time_col else False

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

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

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
        cbar_labels = trj[time_col].iloc[indices].dt.total_seconds().values
        cbar_labels = ["%.2f" % number for number in cbar_labels]
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


def _label_axes(trj: TrajaDataFrame, ax):
    if "spatial_units" in trj.__dict__:
        ax.set_xlabel(trj.spatial_units)
        ax.set_ylabel(trj.spatial_units)
    return ax


def plot_quiver(
    trj: TrajaDataFrame,
    bins: Optional[Union[int, tuple]] = None,
    quiverplot_kws: dict = {},
):
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
    ax.set_aspect("equal")

    plt.show()
    return ax


def plot_contour(
    trj: TrajaDataFrame,
    bins: Optional[Union[int, tuple]] = None,
    filled: bool = True,
    quiver: bool = True,
    contourplot_kws: dict = {},
    contourfplot_kws: dict = {},
    quiverplot_kws: dict = {},
):
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
        cbar = plt.colorbar(cfp, ax=ax)
    cp = plt.contour(
        X, Y, Z, colors="k", linewidths=1, linestyles="solid", **contourplot_kws
    )
    if quiver:
        qp = ax.quiver(X, Y, U, V, units="width", **quiverplot_kws)

    ax = _label_axes(trj, ax)
    ax.set_aspect("equal")

    plt.show()
    return ax


def plot_surface(
    trj: TrajaDataFrame,
    bins: Optional[Union[int, tuple]] = None,
    cmap: str = "jet",
    **surfaceplot_kws: dict,
):
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
    ax = fig.gca(projection="3d")
    ax.plot_surface(
        X, Y, Z, cmap=matplotlib.cm.coolwarm, linewidth=0, **surfaceplot_kws
    )

    ax = _label_axes(trj, ax)

    plt.show()
    return ax


def plot_stream(
    trj: TrajaDataFrame,
    bins: Optional[Union[int, tuple]] = None,
    cmap: str = "jet",
    contourfplot_kws: dict = {},
    contourplot_kws: dict = {},
    streamplot_kws: dict = {},
):
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
    cp = plt.contour(
        X, Y, Z, colors="k", linewidths=1, linestyles="solid", **contourplot_kws
    )
    sp = ax.streamplot(X, Y, U, V, color=Z, cmap=cmap, **streamplot_kws)

    ax = _label_axes(trj, ax)
    ax.set_aspect("equal")

    plt.show()
    return ax


def plot_flow(
    trj: TrajaDataFrame,
    kind: str = "quiver",
    *args,
    contourplot_kws: dict = {},
    contourfplot_kws: dict = {},
    streamplot_kws: dict = {},
    quiverplot_kws: dict = {},
    surfaceplot_kws: dict = {},
):
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
    if kind is "quiver":
        return plot_quiver(trj, *args, **quiverplot_kws)
    elif kind is "contour":
        return plot_contour(trj, filled=False, *args, **quiverplot_kws)
    elif kind is "contourf":
        return plot_contour(trj, *args, **quiverplot_kws)
    elif kind is "stream":
        return plot_stream(
            trj,
            *args,
            contourplot_kws=contourplot_kws,
            contourfplot_kws=contourfplot_kws,
            streamplot_kws=streamplot_kws,
        )
    elif kind is "surface":
        return plot_surface(trj, *args, **surfaceplot_kws)
    else:
        raise NotImplementedError(f"Kind {kind} is not implemented.")


def _get_after_plot_args(**kwargs: dict):
    after_plot_args = dict(
        show=kwargs.pop("show", True), filepath=kwargs.pop("save", None)
    )
    return after_plot_args, kwargs


def trip_grid(
    trj: TrajaDataFrame,
    bins: Union[tuple, int] = 10,
    log: bool = False,
    spatial_units: str = None,
    normalize: bool = False,
    hist_only: bool = False,
    **kwargs,
):
    """Generate a heatmap of time spent by point-to-cell gridding.

    Args:
      bins (int, optional): Number of bins (Default value = 10)
      log (bool): log scale histogram (Default value = False)
      spatial_units (str): units for plotting
      normalize (bool): normalize histogram into density plot
      hist_only (bool): return histogram without plotting

    Returns:
        hist (:class:`numpy.ndarray`): 2D histogram as array
        image (:class:`matplotlib.collections.PathCollection`: image of histogram

    """
    after_plot_args, kwargs = _get_after_plot_args(**kwargs)

    bins = traja.trajectory._bins_to_tuple(trj, bins)
    # TODO: Add kde-based method for line-to-cell gridding
    df = trj[["x", "y"]].dropna()

    # Set aspect if `xlim` and `ylim` set.
    xlim, ylim = traja.trajectory._get_xylim(df)
    xmin, xmax = xlim
    ymin, ymax = ylim

    x, y = zip(*df.values)
    # FIXME: Remove redundant histogram calculation
    hist, x_edges, y_edges = np.histogram2d(
        x, y, bins, range=((xmin, xmax), (ymin, ymax)), normed=normalize
    )

    if log:
        hist = np.log(hist + np.e)
    if hist_only:  # TODO: Evaluate potential use cases or remove
        return (hist, None)
    fig, ax = plt.subplots()

    image = ax.imshow(
        hist, interpolation="bilinear", aspect="equal", extent=[xmin, xmax, ymin, ymax]
    )
    # TODO: Adjust colorbar ytick_labels to correspond with time
    label = "Frames" if not log else "$ln(frames)$"
    plt.colorbar(image, ax=ax, label=label)

    _label_axes(trj, ax)

    plt.title("Time spent{}".format(" (Logarithmic)" if log else ""))

    _process_after_plot_args(**after_plot_args)
    # TODO: Add method for most common locations in grid
    # peak_index = unravel_index(hist.argmax(), hist.shape)
    return hist, image


def _process_after_plot_args(**after_plot_args):
    filepath = after_plot_args.get("filepath")
    if filepath:
        plt.savefig(filepath)
    if after_plot_args.get("show"):
        plt.show()


def _polar_bar(
    radii: np.ndarray,
    theta: np.ndarray,
    bin_size=2,
    ax=None,
    overlap=True,
    **kwargs: str,
):
    after_plot_args, kwargs = _get_after_plot_args(**kwargs)

    title = kwargs.pop("title", None)
    ax = ax or plt.subplot(111, projection="polar")

    hist, bin_edges = np.histogram(
        theta, bins=np.arange(-180, 180 + bin_size, bin_size)
    )
    centers = np.deg2rad(np.ediff1d(bin_edges) // 2 + bin_edges[:-1])

    radians = np.deg2rad(theta)

    width = np.deg2rad(bin_size)
    angle = radians if overlap else centers
    height = radii if overlap else hist
    max_height = max(height)
    bars = ax.bar(angle, height, width=width, bottom=0.0, **kwargs)
    for h, bar in zip(height, bars):
        bar.set_facecolor(plt.cm.jet(h / max_height))
        bar.set_alpha(0.5)
    ax.set_theta_zero_location("N")
    ax.set_xticklabels(["0", "45", "90", "135", "180", "-135", "-90", "-45"])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    _process_after_plot_args(**after_plot_args)

    return ax


def polar_bar(
    trj: TrajaDataFrame,
    feature: str = "turn_angle",
    bin_size: int = 2,
    overlap: bool = True,
    **plot_kws: str,
):
    """Plot polar bar chart.
    Args:
        trj
        bins (int)

    Returns:
        ax

    """
    DIST_THRESHOLD = 0.005
    # Get displacement

    displacement = traja.trajectory.calc_displacement(trj)
    trj["displacement"] = displacement
    trj = trj[trj.displacement > DIST_THRESHOLD]

    if feature == "turn_angle":
        feature_series = traja.trajectory.calc_turn_angle(trj)
        trj["turn_angle"] = feature_series
        trj.loc["turn_angle"] = trj.turn_angle.shift(-1)
    elif feature == "heading":
        feature_series = traja.trajectory.calc_heading(trj)
        trj[feature] = feature_series

    trj = trj[pd.notnull(trj[feature])]
    trj = trj[pd.notnull(trj.displacement)]

    # df = df[["x", "y"]]

    # xy = df[["x", "y"]].values
    # radii, theta = traja.trajectory.cartesian_to_polar(xy)
    ax = _polar_bar(
        trj.displacement, trj[feature], bin_size=bin_size, overlap=overlap, **plot_kws
    )
    return ax


def animate(trj: TrajaDataFrame, polar: bool = True):
    """Animate trajectory.

    Args:
        polar (bool):
    Returns:


    """
    from matplotlib.animation import FuncAnimation

    displacement = traja.trajectory.calc_displacement(trj)
    # heading = traja.calc_heading(trj)
    turn_angle = traja.trajectory.calc_turn_angle(trj)
    xy = trj[["x", "y"]].values

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(211)
    if polar:
        ax2 = plt.subplot(212, polar="projection")

    def colfunc(val, minval, maxval, startcolor, stopcolor):
        """ Convert value in the range minval...maxval to a color in the range
            startcolor to stopcolor. The colors passed and the one returned are
            composed of a sequence of N component values (e.g. RGB).
        """
        f = float(val - minval) / (maxval - minval)
        return tuple(f * (b - a) + a for (a, b) in zip(startcolor, stopcolor))

    POLAR_STEPS = XY_STEPS = 20
    DISPLACEMENT_THRESH = 0.25
    bin_size = 2
    overlap = True

    xlim, ylim = traja.trajectory._get_xylim(trj)

    width = np.pi / 24
    alphas = np.linspace(0.1, 1, XY_STEPS)
    rgba_colors = np.zeros((XY_STEPS, 4))
    rgba_colors[:, 0] = 1.0  # red
    rgba_colors[:, 3] = alphas

    for ind, (x, y) in enumerate(xy):
        if not ind > 1 and not ind + 1 < len(xy):
            continue

        ax1.clear()
        if polar:
            ax2.clear()

        prev_steps = max(ind - XY_STEPS, 0)
        color_cnt = len(xy[prev_steps:ind])
        ax1.scatter(
            xy[prev_steps:ind, 0],
            xy[prev_steps:ind, 1],
            marker="o",
            color=rgba_colors[:color_cnt],
        )
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        displacement_str = (
            rf"$\bf{displacement[ind]:.2f}$"
            if displacement[ind] >= DISPLACEMENT_THRESH
            else f"{displacement[ind]:.2f}"
        )

        ax1.set_title(
            f"frame {ind} - distance (cm/0.25s): {displacement_str}\n"
            "x: {x:.2f}, y: {y:.2f}\n"
            "turn_angle: {turn_angle[ind]"
        )

        if polar and ind > 1:
            start_index = max(ind - POLAR_STEPS, 0)

            theta = turn_angle[start_index:ind]
            radii = displacement[start_index:ind]

            hist, bin_edges = np.histogram(
                theta, bins=np.arange(-180, 180 + bin_size, bin_size)
            )
            centers = np.deg2rad(np.ediff1d(bin_edges) // 2 + bin_edges[:-1])

            radians = np.deg2rad(theta)

            width = np.deg2rad(bin_size)
            angle = radians if overlap else centers
            height = radii if overlap else hist
            max_height = max(height)
            bars = ax2.bar(angle, height, width=width, bottom=0.0)
            for idx, (h, bar) in enumerate(zip(height, bars)):
                bar.set_facecolor(plt.cm.jet(h / max_height))
                bar.set_alpha(0.5 + 0.5 * (idx / POLAR_STEPS))
            ax2.set_theta_zero_location("N")
            ax2.set_xticklabels(["0", "45", "90", "135", "180", "-135", "-90", "-45"])

        plt.tight_layout()
        plt.pause(0.01)
        plt.show(block=False)
