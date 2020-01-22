from collections import OrderedDict
from datetime import timedelta
import logging
from typing import Union, Optional, Tuple, List

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib import dates as md
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_datetime_or_timedelta_dtype,
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
)

import traja
from traja.frame import TrajaDataFrame
from traja.trajectory import coords_to_flow


__all__ = [
    "_get_after_plot_args",
    "_label_axes",
    "_polar_bar",
    "_process_after_plot_args",
    "animate",
    "bar_plot",
    "color_dark",
    "fill_ci",
    "find_runs",
    "plot",
    "plot_3d",
    "plot_actogram",
    "plot_autocorrelation",
    "plot_collection",
    "plot_contour",
    "plot_clustermap",
    "plot_flow",
    "plot_quiver",
    "plot_periodogram",
    "plot_stream",
    "plot_surface",
    "plot_transition_graph",
    "plot_transition_matrix",
    "plot_xy",
    "polar_bar",
    "predict",
    "sans_serif",
    "stylize_axes",
    "trip_grid",
]

logger = logging.getLogger("traja")


def stylize_axes(ax):
    """Add top and right border to plot, set ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_tick_params(top="off", direction="out", width=1)
    ax.yaxis.set_tick_params(right="off", direction="out", width=1)


def sans_serif():
    """Convenience function for changing plot text to serif font."""
    plt.rc("font", family="serif")


def _rolling(df, window, step):
    count = 0
    df_length = len(df)
    while count < (df_length - window):
        yield count, df[count : window + count]
        count += step


def predict(
    xy: np.ndarray,
    nb_steps: int = 10,
    epochs: int = 1000,
    batch_size: int = 1,
    model="lstm",
):  # pragma: no cover
    """Method for training and visualizing LSTM with trajectory datasets."""
    if model is "lstm":
        from traja.models.nn import TrajectoryLSTM

        TrajectoryLSTM(xy, nb_steps=nb_steps, epochs=epochs, batch_size=batch_size)


def bar_plot(trj: TrajaDataFrame, bins: Union[int, tuple] = None, **kwargs) -> Axes:
    """Plot trajectory for single animal over period.

    Args:
      trj (:class:`traja.TrajaDataFrame`): trajectory
      bins (int or tuple): number of bins for x and y
      **kwargs: additional keyword arguments to :meth:`mpl_toolkits.mplot3d.Axed3D.plot`

    Returns:
        ax (:class:`~matplotlib.collections.PathCollection`): Axes of plot

    """
    # TODO: Add time component
    from mpl_toolkits.mplot3d import Axes3D

    bins = traja.trajectory._bins_to_tuple(trj, bins)

    X, Y, U, V = coords_to_flow(trj, bins)

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

    return ax


def plot_rolling_hull(trj: TrajaDataFrame, window=100, step=20, areas=False, **kwargs):
    """Plot rolling convex hull of trajectory. If `areas` is True, only
    areas over time is plotted.
    
    """
    hulls = []

    for offset, window in _rolling(trj, window=window, step=step):
        if window.dropna().empty:
            continue
        shape = window.traja.to_shapely()
        hull = shape.convex_hull
        hulls.append(hull)

    if areas:
        hull_areas = []
        for idx, hull in enumerate(hulls):
            hull_areas.append(hull.area)
        plt.plot(hull_areas, **kwargs)
        plt.title(f"Rolling Trajectory Convex Hull Area\nWindow={window},Step={step}")
        plt.ylabel(f"Area {trj.__dict__.get('spatial_units','m')}")
        plt.xlabel("Frame")
    else:
        xlim, ylim = traja.trajectory._get_xylim(trj)
        plt.xlim = xlim
        plt.ylim = ylim
        for idx, hull in enumerate(hulls):
            if hasattr(
                hull, "exterior"
            ):  # Occassionally a Point object without it reaches
                plt.plot(*hull.exterior.xy, alpha=idx / len(hulls), c="k", **kwargs)
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.set(
            xlabel=f"x ({trj.__dict__.get('spatial_units','m')})",
            ylabel=f"y ({trj.__dict__.get('spatial_units','m')})",
            title="Rolling Trajectory Convex Hull\nWindow={window},Step={step}",
        )


def plot_period(trj: TrajaDataFrame, col="x", dark=(7, 19), **kwargs):
    time_col = traja._get_time_col(trj)
    _trj = trj.set_index(time_col)
    if not col in _trj:
        raise ValueError(f"{col} not a column in dataframe")
    series = _trj[col]
    fig, ax = plt.subplots()
    series.plot(ax=ax)

    dates = np.unique(series.index.date)

    nights = []
    nights.append([(date, date + timedelta(hours=dark[0])) for date in dates])
    nights.append(
        [(date + timedelta(hours=dark[1]), date + timedelta(days=1)) for date in dates]
    )
    for interval in nights:
        t0, t1 = interval
        ax.axvspan(t0, t1, color="gray", alpha=0.2)

    # Format date displayed on the x axis
    xfmt = md.DateFormatter("%H:%M\n%m-%d-%y")
    ax.xaxis.set_major_formatter(xfmt)

    if kwargs.get("interactive"):
        plt.show()


def plot_rolling_hull_3d(trj: TrajaDataFrame, window=100, step=20, **kwargs):
    from mpl_toolkits.mplot3d import Axes3D

    hulls = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for offset, wind in _rolling(trj, window=window, step=step):
        if wind.dropna().empty:
            continue
        shape = wind.traja.to_shapely()
        hull = shape.convex_hull
        hulls.append(hull)

    xlim, ylim = traja.trajectory._get_xylim(trj)
    plt.xlim = xlim
    plt.ylim = ylim
    outlines = []
    for idx, hull in enumerate(hulls):
        if hasattr(hull, "exterior"):  # Occassionally a Point object without it reaches
            outlines.append(np.array(hull.exterior.xy))

    # Add plots to axes
    NLINES = len(outlines)
    cm = plt.get_cmap(kwargs.get("cmap", "plasma"))
    ax.set_prop_cycle(color=[cm(1.0 * i / (NLINES)) for i in range(NLINES)])
    for z, xy in enumerate(outlines):
        ax.plot(*xy, z)

    ax.set(
        xlabel=f"{trj.__dict__.get('spatial_units','m')}",
        ylabel=f"{trj.__dict__.get('spatial_units','m')}",
        title=f"Rolling Trajectory Convex Hull\nWindow={window},Step={step}",
    )

    if kwargs.get("interactive"):
        plt.show()


def plot_3d(trj: TrajaDataFrame, **kwargs) -> matplotlib.collections.PathCollection:
    """Plot 3D trajectory for single identity over period.

    Args:
      trj (:class:`traja.TrajaDataFrame`): trajectory
      n_coords (int, optional): Number of coordinates to plot
      **kwargs: additional keyword arguments to :meth:`matplotlib.axes.Axes.scatter`

    Returns:
        ax (:class:`~matplotlib.collections.PathCollection`): Axes of plot

    .. note::
        Takes a while to plot large trajectories. Consider using first::
        
            rt = trj.traja.rediscretize(R=1.) # Replace R with appropriate step length
            rt.traja.plot_3d()

    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x", fontsize=15)
    ax.set_zlabel("time", fontsize=15)
    ax.set_ylabel("y", fontsize=15)
    title = kwargs.pop("title", "Trajectory")
    ax.set_title(f"{title}", fontsize=20)
    ax.plot(trj.x, trj.y, trj.index)
    cmap = kwargs.pop("cmap", "winter")
    cm = plt.get_cmap(cmap)
    NPOINTS = len(trj)
    ax.set_prop_cycle(color=[cm(1.0 * i / (NPOINTS - 1)) for i in range(NPOINTS - 1)])
    for i in range(NPOINTS - 1):
        ax.plot(trj.x[i : i + 2], trj.y[i : i + 2], trj.index[i : i + 2])

    dist = kwargs.pop("dist", None)
    if dist:
        ax.dist = dist
    labelpad = kwargs.pop("labelpad", None)
    if labelpad:
        from matplotlib import rcParams

        rcParams["axes.labelpad"] = labelpad

    return ax


def plot(
    trj: TrajaDataFrame,
    n_coords: Optional[int] = None,
    show_time: bool = False,
    accessor: Optional[traja.TrajaAccessor] = None,
    ax=None,
    **kwargs,
) -> matplotlib.collections.PathCollection:
    """Plot trajectory for single animal over period.

    Args:
      trj (:class:`traja.TrajaDataFrame`): trajectory
      n_coords (int, optional): Number of coordinates to plot
      show_time (bool): Show colormap as time
      accessor (:class:`~traja.accessor.TrajaAccessor`, optional): TrajaAccessor instance
      ax (:class:`~matplotlib.axes.Axes`): axes for plotting
      interactive (bool): show plot immediately
      **kwargs: additional keyword arguments to :meth:`matplotlib.axes.Axes.scatter`

    Returns:
        collection (:class:`~matplotlib.collections.PathCollection`): collection that was plotted

    """
    import matplotlib.patches as patches
    from matplotlib.path import Path

    after_plot_args, kwargs = _get_after_plot_args(**kwargs)

    GRAY = "#999999"

    xlim = kwargs.pop("xlim", None)
    ylim = kwargs.pop("ylim", None)
    if not xlim or not ylim:
        xlim, ylim = traja.trajectory._get_xylim(trj)

    title = kwargs.pop("title", None)
    time_units = kwargs.pop("time_units", "s")
    fps = kwargs.pop("fps", None)
    figsize = kwargs.pop("figsize", None)

    coords = trj[["x", "y"]]
    time_col = traja.trajectory._get_time_col(trj)

    if time_col == "index":
        is_datetime = True
    else:
        is_datetime = is_datetime64_any_dtype(trj[time_col]) if time_col else False

    if n_coords is None:
        # Plot all coords
        start, end = 0, len(coords)
        verts = coords.iloc[start:end].values
    else:
        # Plot first `n_coords`
        verts = coords.iloc[:n_coords].values

    n_coords = len(verts)

    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    path = Path(verts, codes)

    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
        fig.canvas.draw()

    patch = patches.PathPatch(path, edgecolor=GRAY, facecolor="none", lw=3, alpha=0.3)
    ax.add_patch(patch)

    xs, ys = zip(*verts)

    if time_col == "index":
        # DatetimeIndex determines color
        colors = [ind for ind, x in enumerate(trj.index[:n_coords])]
    elif time_col and time_col != "index":
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

    collection = ax.scatter(
        xs,
        ys,
        c=colors,
        s=kwargs.pop("s", 1),
        cmap=plt.cm.viridis,
        alpha=0.7,
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if kwargs.pop("invert_yaxis", None):
        plt.gca().invert_yaxis()

    _label_axes(trj, ax)
    ax.set_title(title)
    ax.set_aspect("equal")

    # Number of color bar ticks
    CBAR_TICKS = 10 if n_coords > 20 else n_coords
    indices = np.linspace(0, n_coords - 1, CBAR_TICKS, endpoint=True, dtype=int)
    cbar = plt.colorbar(
        collection, fraction=0.046, pad=0.04, orientation="vertical", label=label
    )

    # Get colorbar labels from time
    if time_col == "index":
        if is_datetime64_any_dtype(trj.index):
            cbar_labels = (
                trj.index[indices].strftime("%Y-%m-%d %H:%M:%S").values.astype(str)
            )
        elif is_timedelta64_dtype(trj.index):
            if time_units in ("s", "", None):
                cbar_labels = [round(x, 2) for x in trj.index[indices].total_seconds()]
            else:
                logger.error("Time unit {} not yet implemented".format(time_units))
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
        if time_col:
            cbar_labels = trj[time_col][indices].values
        else:
            cbar_labels = trj.index[indices].values
        cbar_labels = np.round(cbar_labels, 6)
        if fps is not None and fps > 0 and fps is not 1 and show_time:
            cbar_labels = cbar_labels / fps

    cbar.set_ticks(indices)
    cbar.set_ticklabels(cbar_labels)
    plt.tight_layout()

    _process_after_plot_args(**after_plot_args)
    return collection


def plot_periodogram(trj, coord: str = "y", fs: int = 1, interactive: bool = True):
    """Plot power spectral density using a periodogram.

    Args:
        trj - Trajectory
        coord - choice of 'x' or 'y'
        fs - Sampling frequency
        interactive - Plot immediately
    
    Returns:
        Figure

    """
    from scipy import signal

    vals = trj[coord].values
    f, Pxx = signal.periodogram(vals, fs=fs, window="hanning", scaling="spectrum")
    plt.title("Power Spectrum")
    if interactive:
        plt.plot(f, Pxx)
    return plt.gcf()


def plot_autocorrelation(
    trj: TrajaDataFrame,
    coord: str = "y",
    unit: str = "Days",
    sample_rate: float = 3.0,
    xmax: int = 1000,
    interactive: bool = True,
):
    """Plot autocorrelation of given coordinate.
    
    Args:
        trj - Trajectory
        coord - 'x' or 'y'
        unit - string, eg, 'Days'
        sample_rate - sample rate
        xmax - max xaxis value
        interactive - Plot immediately
    
    Returns:
        Matplotlib Figure
    
    """
    from statsmodels import api as sm

    pos = trj[coord].values
    acf = sm.tsa.acf(pos, nlags=len(pos))
    lag = np.arange(len(pos)) / sample_rate
    plt.plot(lag, acf)
    plt.xlim((0, xmax))
    plt.xlabel(f"Lags ({unit})")
    plt.ylabel("Autocorrelation")
    if interactive:
        plt.show()
    return plt.gcf()


def plot_collection(
    trjs: Union[pd.DataFrame, TrajaDataFrame],
    id_col: str = "id",
    colors: Optional[Union[dict, List[str]]] = None,
    **kwargs,
):
    """Plot trajectories of multiple subjects identified by `id`.

    Args:
        trjs: dataframe with multiple trajectories
        id_col: name of id_col, default is "id"
        colors (Optional): color lookup matching substrings to discreet colors. Possible values are, eg:
                                - {"car0":"red","car1":"blue"}
                                - {"car":"red","person":blue"}
                                - ["car", "person"]
        kwargs: kwargs to :meth:`matplotlib.axes.Axes.plot`

    Returns:
        lines (list of `~matplotlib.lines.Line2D` objects): lines of plot

    """
    ids = trjs[id_col].unique()

    # Get plot keyword args
    colormap = kwargs.pop("cmap", "hsv")
    alpha = kwargs.pop("alpha", 0.2)
    linestyle = kwargs.pop("linestyle", "-")
    marker = kwargs.pop("marker", "o")

    labels = [None] * len(ids)

    if not colors:
        cmap = plt.cm.get_cmap(colormap, lut=len(ids) if len(ids) > 1 else None)
        colors = [cmap(idx) for idx in range(len(ids))]
    elif isinstance(colors, list):
        cmap = plt.cm.get_cmap(colormap, len(colors))
        color_lookup = []
        for ind, id in enumerate(ids):
            for idx, substring in enumerate(colors):
                if substring in id:
                    color_lookup.append(cmap(idx))
                    labels[ind] = substring
                    break
            else:
                raise Exception(f"No substring matching {id} in {colors}.")
        colors = color_lookup
    elif isinstance(colors, dict):
        color_lookup = [colors.get(id) for id in ids]
        colors = color_lookup
        labels = ids

    fig, ax = plt.subplots()
    lines = []
    for idx, id in enumerate(ids):
        trj = trjs[trjs[id_col] == id]
        l = ax.plot(
            trj.x,
            trj.y,
            linestyle=linestyle,
            marker=marker,
            c=colors[idx],
            alpha=alpha,
            label=labels[idx],
            **kwargs,
        )
        lines.extend(l)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
    )
    plt.tight_layout()
    return lines


def _label_axes(trj: TrajaDataFrame, ax) -> Axes:
    if "spatial_units" in trj.__dict__:
        ax.set_xlabel(trj.__dict__.get("spatial_units", "m"))
        ax.set_ylabel(trj.__dict__.get("spatial_units", "m"))
    return ax


def plot_quiver(
    trj: TrajaDataFrame,
    bins: Optional[Union[int, tuple]] = None,
    quiverplot_kws: dict = {},
    **kwargs,
) -> Axes:
    """Plot average flow from each grid cell to neighbor.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        quiverplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.quiver`

    Returns:
        ax (:class:`~matplotlib.axes.Axes`): Axes of quiver plot
    """

    after_plot_args, _ = _get_after_plot_args(**kwargs)

    X, Y, U, V = coords_to_flow(trj, bins)
    Z = np.sqrt(U * U + V * V)

    fig, ax = plt.subplots()

    ax.quiver(X, Y, U, V, units="width", **quiverplot_kws)
    ax = _label_axes(trj, ax)
    ax.set_aspect("equal")

    _process_after_plot_args(**after_plot_args)
    return ax


def plot_contour(
    trj: TrajaDataFrame,
    bins: Optional[Union[int, tuple]] = None,
    filled: bool = True,
    quiver: bool = True,
    contourplot_kws: dict = {},
    contourfplot_kws: dict = {},
    quiverplot_kws: dict = {},
    **kwargs,
) -> Axes:
    """Plot average flow from each grid cell to neighbor.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        contourplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contour`
        contourfplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contourf`
        quiverplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.quiver`

    Returns:
        ax (:class:`~matplotlib.axes.Axes`): Axes of quiver plot
    """

    after_plot_args, _ = _get_after_plot_args(**kwargs)

    X, Y, U, V = coords_to_flow(trj, bins)
    Z = np.sqrt(U * U + V * V)

    fig, ax = plt.subplots()

    if filled:
        cfp = plt.contourf(X, Y, Z, **contourfplot_kws)
        plt.colorbar(cfp, ax=ax)
    plt.contour(
        X, Y, Z, colors="k", linewidths=1, linestyles="solid", **contourplot_kws
    )
    if quiver:
        ax.quiver(X, Y, U, V, units="width", **quiverplot_kws)

    ax = _label_axes(trj, ax)
    ax.set_aspect("equal")

    _process_after_plot_args(**after_plot_args)
    return ax


def plot_surface(
    trj: TrajaDataFrame,
    bins: Optional[Union[int, tuple]] = None,
    cmap: str = "jet",
    **surfaceplot_kws: dict,
) -> Figure:
    """Plot surface of flow from each grid cell to neighbor in 3D.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        cmap (str): color map
        surfaceplot_kws: Additional keyword arguments for :meth:`~mpl_toolkits.mplot3D.Axes3D.plot_surface`

    Returns:
        ax (:class:`~matplotlib.axes.Axes`): Axes of quiver plot
    """
    from mpl_toolkits.mplot3d import Axes3D

    after_plot_args, surfaceplot_kws = _get_after_plot_args(**surfaceplot_kws)

    X, Y, U, V = coords_to_flow(trj, bins)
    Z = np.sqrt(U * U + V * V)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(
        X, Y, Z, cmap=matplotlib.cm.coolwarm, linewidth=0, **surfaceplot_kws
    )

    ax = _label_axes(trj, ax)
    try:
        ax.set_aspect("equal")
    except NotImplementedError:
        # 3D
        pass

    _process_after_plot_args(**after_plot_args)
    return ax


def plot_stream(
    trj: TrajaDataFrame,
    bins: Optional[Union[int, tuple]] = None,
    cmap: str = "jet",
    contourfplot_kws: dict = {},
    contourplot_kws: dict = {},
    streamplot_kws: dict = {},
    **kwargs,
) -> Figure:
    """Plot average flow from each grid cell to neighbor.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        contourplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contour`
        contourfplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contourf`
        streamplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.streamplot`

    Returns:
        ax (:class:`~matplotlib.axes.Axes`): Axes of stream plot

    """

    after_plot_args, _ = _get_after_plot_args(**kwargs)
    X, Y, U, V = coords_to_flow(trj, bins)
    Z = np.sqrt(U * U + V * V)

    fig, ax = plt.subplots()

    plt.contourf(X, Y, Z, **contourfplot_kws)
    plt.contour(
        X, Y, Z, colors="k", linewidths=1, linestyles="solid", **contourplot_kws
    )
    ax.streamplot(X, Y, U, V, color=Z, cmap=cmap, **streamplot_kws)

    ax = _label_axes(trj, ax)
    ax.set_aspect("equal")

    _process_after_plot_args(**after_plot_args)
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
    **kwargs,
) -> Figure:
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
        ax (:class:`~matplotlib.axes.Axes`): Axes of plot
    """
    if kind == "quiver":
        return plot_quiver(trj, *args, **quiverplot_kws, **kwargs)
    elif kind == "contour":
        return plot_contour(trj, filled=False, *args, **quiverplot_kws, **kwargs)
    elif kind == "contourf":
        return plot_contour(trj, *args, **quiverplot_kws, **kwargs)
    elif kind == "stream":
        return plot_stream(
            trj,
            *args,
            contourplot_kws=contourplot_kws,
            contourfplot_kws=contourfplot_kws,
            streamplot_kws=streamplot_kws,
            **kwargs,
        )
    elif kind == "surface":
        return plot_surface(trj, *args, **surfaceplot_kws, **kwargs)
    else:
        raise NotImplementedError(f"Kind {kind} is not implemented.")


def _get_after_plot_args(**kwargs: dict) -> (dict, dict):
    after_plot_args = dict(
        interactive=kwargs.pop("interactive", True),
        filepath=kwargs.pop("filepath", None),
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
) -> Tuple[np.ndarray, PathCollection]:
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
    if "xlim" in kwargs and "ylim" in kwargs:
        xlim, ylim = kwargs.pop("xlim"), kwargs.pop("ylim")
    else:
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


def color_dark(
    series: pd.Series, ax: matplotlib.axes.Axes, start: int = 19, end: int = 7
):
    """Color dark phase in plot."""
    assert is_datetime_or_timedelta_dtype(
        series.index
    ), f"Series must have datetime index but has {type(series.index)}"

    dark_mask = (series.index.hour >= start) | (series.index.hour < end)
    run_values, run_starts, run_lengths = find_runs(dark_mask)

    for idx, is_dark in enumerate(run_values):
        if is_dark:
            start = run_starts[idx]
            end = run_starts[idx] + run_lengths[idx] - 1
            ax.axvspan(series.index[start], series.index[end], alpha=0.5, color="gray")


def find_runs(x: pd.Series) -> (np.ndarray, np.ndarray, np.ndarray):
    """Find runs of consecutive items in an array.
    From https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def fill_ci(series: pd.Series, window: Union[int, str]) -> Figure:
    """Fill confidence interval defined by SEM over mean of `window`. Window can be interval or offset, eg, '30s'."""
    assert is_datetime_or_timedelta_dtype(
        series.index
    ), f"Series index must be datetime but is {type(series.index)}"
    smooth_path = series.rolling(window).mean()
    path_deviation = series.rolling(window).std()

    fig, ax = plt.subplots()

    plt.plot(smooth_path.index, smooth_path, "b")
    plt.fill_between(
        path_deviation.index,
        (smooth_path - 2 * path_deviation),
        (smooth_path + 2 * path_deviation),
        color="b",
        alpha=0.2,
    )

    plt.gcf().autofmt_xdate()
    return ax


def plot_xy(xy: np.ndarray, *args: Optional, **kwargs: Optional):
    """Plot trajectory from xy values.

    Args:

        xy (np.ndarray) : xy values of dimensions N x 2
        *args           : Plot args
        **kwargs        : Plot kwargs
    """
    trj = traja.from_xy(xy)
    trj.traja.plot(*args, **kwargs)


def plot_actogram(
    series: pd.Series, dark=(19, 7), ax: matplotlib.axes.Axes = None, **kwargs
):
    """Plot activity or displacement as an actogram.

    .. note::

       For published example see Eckel-Mahan K, Sassone-Corsi P. Phenotyping Circadian Rhythms in Mice.
       Curr Protoc Mouse Biol. 2015;5(3):271-281. Published 2015 Sep 1. doi:10.1002/9780470942390.mo140229

    """
    after_plot_args, _ = _get_after_plot_args(**kwargs)
    assert is_datetime_or_timedelta_dtype(
        series.index
    ), f"Series must have datetime index but has {type(series.index)}"

    ax = series.plot(ax=ax)
    ax.set_ylabel(series.name)

    color_dark(series, ax, start=dark[0], end=dark[1])

    _process_after_plot_args(**after_plot_args)


def _polar_bar(
    radii: np.ndarray,
    theta: np.ndarray,
    bin_size: int = 2,
    ax: Optional[matplotlib.axes.Axes] = None,
    overlap: bool = True,
    **kwargs: str,
) -> Axes:
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
    if isinstance(ax, matplotlib.axes.Axes):
        ax.set_theta_zero_location("N")
        ax.set_xticklabels(["0", "45", "90", "135", "180", "-135", "-90", "-45"])
    if title:
        plt.title(title + "\n", y=1.08)
    plt.tight_layout()

    _process_after_plot_args(**after_plot_args)
    return ax


def polar_bar(
    trj: TrajaDataFrame,
    feature: str = "turn_angle",
    bin_size: int = 2,
    threshold: float = 0.001,
    overlap: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kws: str,
) -> Axes:
    """Plot polar bar chart.

    Args:
        trj (:class:`traja.TrajaDataFrame`): trajectory
        feature (str): Options: 'turn_angle', 'heading'
        bin_size (int): width of bins
        threshold (float): filter for step distance
        overlap (bool): Overlapping shows all values, if set to false is a histogram

    Returns:
        ax (:class:`~matplotlib.collections.PathCollection`): Axes of plot

    """
    # Get displacement
    displacement = traja.trajectory.calc_displacement(trj)
    trj["displacement"] = displacement
    trj = trj.loc[trj.displacement > threshold]
    if feature == "turn_angle":
        feature_series = traja.trajectory.calc_turn_angle(trj)
        trj["turn_angle"] = feature_series
        trj.turn_angle = trj.turn_angle.shift(-1)
    elif feature == "heading":
        feature_series = traja.trajectory.calc_heading(trj)
        trj[feature] = feature_series

    trj = trj[pd.notnull(trj[feature])]
    trj = trj[pd.notnull(trj.displacement)]

    assert (
        len(trj) > 0
    ), f"Dataframe is empty after filtering for step distance threshold {threshold}"

    ax = _polar_bar(
        trj.displacement,
        trj[feature],
        bin_size=bin_size,
        overlap=overlap,
        ax=ax,
        **plot_kws,
    )
    return ax


def plot_clustermap(
    displacements: List[pd.Series],
    rule: Optional[str] = None,
    nr_steps=None,
    colors: Optional[List[Union[int, str]]] = None,
    **kwargs,
):
    """Plot cluster map / dendrogram of trajectories with DatetimeIndex.

    Args:
        displacements: list of pd.Series, outputs of :func:`traja.calc_displacement()`
        rule:   how to resample series, eg '30s' for 30-seconds
        nr_steps: select first N samples for clustering
        colors: list of colors (eg, 'b','r') to map to each trajectory
        kwargs: keyword arguments for :func:`seaborn.clustermap`

    Returns:
        cg: a :func:`seaborn.matrix.ClusterGrid` instance

    .. note::

        Requires seaborn to be installed. Install it with 'pip install seaborn'.

    """
    try:
        import seaborn as sns
    except ImportError:
        logging.error("seaborn is not installed. Install it with 'pip install seaborn'")
        return

    after_plot_args, _ = _get_after_plot_args(**kwargs)

    series_lst = []
    for disp in displacements:
        if rule:
            disp = disp.resample(rule).sum()
        series_lst.append(disp)

    df = pd.DataFrame(series_lst)
    df.columns = range(len(df.columns))
    df.reset_index(drop=True, inplace=True)

    if not nr_steps:
        nr_steps = df.shape[1]

    cg = sns.clustermap(
        df.fillna(0).iloc[:, :nr_steps],
        xticklabels=False,
        col_cluster=False,
        figsize=(16, 6),
        cmap="Greys",
        row_colors=colors,
        **kwargs,
    )
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    _process_after_plot_args(**after_plot_args)
    return cg


def _get_markov_edges(Q: pd.DataFrame, greater_than=0.1):
    """Select edges greater than a threshold of weight."""
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            if greater_than and Q.loc[idx, col] > greater_than:
                edges[(idx, col)] = Q.loc[idx, col]
    return edges


def plot_transition_graph(
    data: Union[pd.DataFrame, traja.TrajaDataFrame, np.ndarray],
    outpath="markov.dot",
    interactive=True,
):
    """Plot transition graph with networkx.

    Args:
        data (trajectory or transition_matrix)
    
    .. note::        
        Modified from http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017
    
    """
    try:
        import networkx as nx
        import pydot
        import graphviz
    except ImportError as e:
        raise ImportError(f"{e} - please install it with pip")

    if (
        isinstance(data, (traja.TrajaDataFrame))
        or isinstance(data, pd.DataFrame)
        and "x" in data
    ):
        transition_matrix = traja.transitions(data)
        edges_wts = _get_markov_edges(pd.DataFrame(transition_matrix))
        states_ = list(range(transition_matrix.shape[0]))

    # create graph object
    G = nx.MultiDiGraph()

    # nodes correspond to states
    G.add_nodes_from(states_)

    # edges represent transition probabilities
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v.round(4), label=v.round(4))

    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
    nx.draw_networkx(G, pos)

    # create edge labels for jupyter plot but is not necessary
    edge_labels = {(n1, n2): d["label"] for n1, n2, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    if os.exists(outpath):
        logging.info(f"Overwriting {outpath}")
    nx.drawing.nx_pydot.write_dot(G, outpath)

    if interactive:
        # Plot
        from graphviz import Source

        s = Source.from_file(outpath)
        s.view()


def plot_transition_matrix(
    data: Union[pd.DataFrame, traja.TrajaDataFrame, np.ndarray],
    interactive=True,
    **kwargs,
) -> matplotlib.image.AxesImage:
    """Plot transition matrix.
    
    Args:
        data (trajectory or square transition matrix)
        interactive (bool): show plot
        kwargs: kwargs to :func:`traja.grid_coordinates`

    Returns:
        axesimage (matplotlib.image.AxesImage)
    
    """
    if isinstance(data, np.ndarray):
        if data.shape[0] != data.shape[1]:
            raise ValueException(
                f"Ndarray input must be square transition matrix, shape is {data.shape}"
            )
        transition_matrix = data
    elif isinstance(data, (pd.DataFrame, traja.TrajaDataFrame)):
        transition_matrix = traja.transitions(data, **kwargs)
    img = plt.imshow(transition_matrix)
    if interactive:
        plt.show()
    return img


def animate(trj: TrajaDataFrame, polar: bool = True, save: bool = False):
    """Animate trajectory.

    Args:
        polar (bool): include polar bar chart with turn angle
        save (bool): save video to ``trajectory.mp4``

    """
    from matplotlib import animation
    from matplotlib.animation import FuncAnimation

    displacement = traja.trajectory.calc_displacement(trj).reset_index(drop=True)
    # heading = traja.calc_heading(trj)
    turn_angle = traja.trajectory.calc_turn_angle(trj).reset_index(drop=True)
    xy = trj[["x", "y"]].reset_index(drop=True)

    POLAR_STEPS = XY_STEPS = 20
    DISPLACEMENT_THRESH = 0.025
    bin_size = 2
    overlap = True

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(211)

    fig.add_subplot(ax1)
    if polar:
        ax2 = plt.subplot(212, polar="projection")
        ax2.set_theta_zero_location("N")
        ax2.set_xticklabels(["0", "45", "90", "135", "180", "-135", "-90", "-45"])
        fig.add_subplot(ax2)
        ax2.bar(
            np.zeros(XY_STEPS), np.zeros(XY_STEPS), width=np.zeros(XY_STEPS), bottom=0.0
        )

    xlim, ylim = traja.trajectory._get_xylim(trj)
    ax1.set(
        xlim=xlim,
        ylim=ylim,
        ylabel=trj.__dict__.get("spatial_units", "m"),
        xlabel=trj.__dict__.get("spatial_units", "m"),
        aspect="equal",
    )

    alphas = np.linspace(0.1, 1, XY_STEPS)
    rgba_colors = np.zeros((XY_STEPS, 4))
    rgba_colors[:, 0] = 1.0  # red
    rgba_colors[:, 3] = alphas
    scat = ax1.scatter(
        range(XY_STEPS), range(XY_STEPS), marker=".", color=rgba_colors[:XY_STEPS]
    )

    def update(frame_number):
        ind = frame_number % len(xy)
        if ind < XY_STEPS:
            scat.set_offsets(xy[:ind])
        else:
            prev_steps = max(ind - XY_STEPS, 0)
            scat.set_offsets(xy[prev_steps:ind])

        displacement_str = (
            rf"$\bf{displacement[ind]:.2f}$"
            if displacement[ind] >= DISPLACEMENT_THRESH
            else f"{displacement[ind]:.2f}"
        )

        x, y = xy.iloc[ind]
        ax1.set_title(
            f"frame {ind} - distance (cm/0.25s): {displacement_str}\n"
            f"x: {x:.2f}, y: {y:.2f}\n"
            f"turn_angle: {turn_angle[ind]:.2f}"
        )

        if polar and ind > 1:
            ax2.clear()
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
            max_height = displacement.max() if overlap else max(hist)

            bars = ax2.bar(angle, height, width=width, bottom=0.0)
            for idx, (h, bar) in enumerate(zip(height, bars)):
                bar.set_facecolor(plt.cm.jet(h / max_height))
                bar.set_alpha(0.8 * (idx / POLAR_STEPS))
            ax2.set_theta_zero_location("N")
            ax2.set_xticklabels(["0", "45", "90", "135", "180", "-135", "-90", "-45"])
        plt.tight_layout()

    anim = FuncAnimation(fig, update, interval=10, frames=range(len(xy)))
    if save:
        try:
            anim.save("trajectory.mp4", writer=animation.FFMpegWriter(fps=10))
        except FileNotFoundError:
            raise Exception("FFmpeg not installed, please install it.")
    else:
        plt.show()
