from typing import Union, Optional, Tuple, List

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
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
):  # pragma: no cover
    """Method for training and visualizing LSTM with trajectory data."""
    if model is "lstm":
        from traja.models.nn import TrajectoryLSTM

        TrajectoryLSTM(xy, nb_steps=nb_steps, epochs=epochs, batch_size=batch_size)


def bar_plot(trj: TrajaDataFrame, bins: Union[int, tuple] = None, **kwargs) -> Axes:
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
    ax=None,
    **kwargs,
) -> Figure:
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
        fig (:class:`~matplotlib.figure.Figure`): Figure of plot

    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
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

    if time_col is "index":
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
    else:
        fig = plt.gcf()

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
        xs,
        ys,
        c=colors,
        s=kwargs.pop("s", 5),
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
        sc, fraction=0.046, pad=0.04, orientation="vertical", label=label
    )

    # Get colorbar labels from time
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

    _process_after_plot_args(**after_plot_args)
    return fig


def _label_axes(trj: TrajaDataFrame, ax) -> Axes:
    if "spatial_units" in trj.__dict__:
        ax.set_xlabel(trj.spatial_units)
        ax.set_ylabel(trj.spatial_units)
    return ax


def plot_quiver(
    trj: TrajaDataFrame,
    bins: Optional[Union[int, tuple]] = None,
    quiverplot_kws: dict = {},
    **kwargs,
) -> Figure:
    """Plot average flow from each grid cell to neighbor.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        quiverplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.quiver`

    Returns:
        fig (:class:`~matplotlib.figure.Figure`): Axes of quiver plot
    """

    after_plot_args, _ = _get_after_plot_args(**kwargs)

    X, Y, U, V = coords_to_flow(trj, bins)
    Z = np.sqrt(U * U + V * V)

    fig, ax = plt.subplots()

    qp = ax.quiver(X, Y, U, V, units="width", **quiverplot_kws)
    ax = _label_axes(trj, ax)
    ax.set_aspect("equal")

    _process_after_plot_args(**after_plot_args)
    return fig


def plot_contour(
    trj: TrajaDataFrame,
    bins: Optional[Union[int, tuple]] = None,
    filled: bool = True,
    quiver: bool = True,
    contourplot_kws: dict = {},
    contourfplot_kws: dict = {},
    quiverplot_kws: dict = {},
    **kwargs,
) -> Figure:
    """Plot average flow from each grid cell to neighbor.

    Args:
        bins (int or tuple): Tuple of x,y bin counts; if `bins` is int, bin count of x,
                                with y inferred from aspect ratio
        contourplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contour`
        contourfplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.contourf`
        quiverplot_kws: Additional keyword arguments for :meth:`~matplotlib.axes.Axes.quiver`

    Returns:
        fig (:class:`~matplotlib.figure.Figure`): Figure of quiver plot
    """

    after_plot_args, _ = _get_after_plot_args(**kwargs)

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

    _process_after_plot_args(**after_plot_args)
    return fig


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
        fig (:class:`~matplotlib.figure.Figure`): Figure of quiver plot
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
    ax.set_aspect("equal")

    _process_after_plot_args(**after_plot_args)
    return fig


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
        fig (:class:`~matplotlib.figure.Figure`): Figure of stream plot

    """

    after_plot_args, _ = _get_after_plot_args(**kwargs)

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

    _process_after_plot_args(**after_plot_args)
    return fig


def plot_flow(
    trj: TrajaDataFrame,
    kind: str = "quiver",
    *args,
    contourplot_kws: dict = {},
    contourfplot_kws: dict = {},
    streamplot_kws: dict = {},
    quiverplot_kws: dict = {},
    surfaceplot_kws: dict = {},
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
        fig (:class:`~matplotlib.figure.Figure`): Figure of plot
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
    if after_plot_args.get("interactive"):
        plt.show()


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
    return fig


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
    overlap: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kws: str,
) -> Axes:
    """Plot polar bar chart.
    Args:
        trj
        feature (str): Options: 'turn_angle', 'heading'
        bins (int): width of bins
        overlap (bool): Overlapping shows all values, if set to false is a histogram

    Returns:
        ax

    """
    DIST_THRESHOLD = 0.001
    # Get displacement

    displacement = traja.trajectory.calc_displacement(trj)
    trj["displacement"] = displacement
    trj = trj.loc[trj.displacement > DIST_THRESHOLD]
    if feature == "turn_angle":
        feature_series = traja.trajectory.calc_turn_angle(trj)
        trj["turn_angle"] = feature_series
        trj.turn_angle = trj.turn_angle.shift(-1)
    elif feature == "heading":
        feature_series = traja.trajectory.calc_heading(trj)
        trj[feature] = feature_series

    trj = trj[pd.notnull(trj[feature])]
    trj = trj[pd.notnull(trj.displacement)]

    assert len(trj) > 0, "Dataframe is empty after filtering, check coordinates"

    ax = _polar_bar(
        trj.displacement,
        trj[feature],
        bin_size=bin_size,
        overlap=overlap,
        ax=ax,
        **plot_kws,
    )
    return ax


def animate(trj: TrajaDataFrame, polar: bool = True, save: bool = False):
    """Animate trajectory.

    Args:
        polar (bool): include polar bar chart with turn angle
        save (bool): save video to ``trajectory.mp4``
    Returns:

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
        bars = ax2.bar(
            np.zeros(XY_STEPS), np.zeros(XY_STEPS), width=np.zeros(XY_STEPS), bottom=0.0
        )

    xlim, ylim = traja.trajectory._get_xylim(trj)
    ax1.set(
        xlim=xlim,
        ylim=ylim,
        ylabel=trj.spatial_units or "m",
        xlabel=trj.spatial_units or "m",
        aspect="equal",
    )

    width = np.pi / 24
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
            scat.set_offsets(xy[:XY_STEPS])
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
