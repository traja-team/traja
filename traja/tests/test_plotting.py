import warnings

import numpy as np
import numpy.testing as npt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import traja

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

df = traja.generate(n=10)


def test_stylize_axes():
    fig = traja.plot(df, interactive=False)
    ax = fig.axes[1]
    traja.plotting.stylize_axes(ax)


def test_color_dark():
    df = traja.generate(n=10)
    index = pd.DatetimeIndex(range(10))
    df.index = index
    traja.plot(df, interactive=False)
    ax = plt.gca()
    traja.color_dark(df.x, ax)


def test_sans_serif():
    traja.plotting.sans_serif()


def test_plot_flow():
    traja.plot_flow(df, interactive=False)


def test_plot_contour():
    fig = traja.plot_contour(df, interactive=False)


def test_plot_surface():
    fig = traja.plot_surface(df, interactive=False)


def test_plot_stream():
    fig = traja.plot_stream(df, interactive=False)


def test_trip_grid():
    traja.plotting.trip_grid(df, interactive=False)


def test_label_axes():
    df.traja.plot(interactive=False)
    ax = plt.gca()
    traja.plotting._label_axes(df, ax)


def test_plot_actogram():
    df = traja.generate(n=10)
    index = pd.DatetimeIndex(range(10))
    df.index = index
    activity = traja.calc_displacement(df)
    activity.name = "activity"
    traja.plotting.plot_actogram(df.x, interactive=False)


def test_plot_xy():
    traja.plotting.plot_xy(df, interactive=False)


def test_polar_bar():
    traja.plotting.polar_bar(df, interactive=False)


def test_find_runs():
    actual = traja.find_runs(df.x)
    expected = (
        np.array(
            [
                0.0,
                1.32337069,
                2.27583754,
                2.27428561,
                0.33602988,
                -1.45569092,
                -3.54444211,
                -5.38659793,
                -7.5085444,
                -9.35325517,
            ]
        ),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    for i in range(len(actual)):
        npt.assert_allclose(actual[i], expected[i])


def test_plot_clustermap():
    trjs = [traja.generate(seed=i) for i in range(20)]

    # Calculate displacement
    displacements = [trj.traja.calc_displacement() for trj in trjs]

    traja.plot_clustermap(displacements)


def test_plot():
    fig = traja.plotting.plot(df, interactive=False)
    ax = fig.axes[1]
    coll = ax.collections[0]
    path = coll.get_paths()[0]
    npt.assert_allclose(
        path._vertices,
        np.array(
            [[0.0, 0.0], [9.0, 0.0], [9.0, 0.03515625], [0.0, 0.03515625], [0.0, 0.0]]
        ),
    )
