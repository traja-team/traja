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
    fig = traja.plot(df)
    ax = fig.axes[1]
    traja.plotting.stylize_axes(ax)


def test_color_dark():
    df = traja.generate(n=10)
    index = pd.DatetimeIndex(range(11))
    df.index = index
    traja.plot(df)
    ax = plt.gca()
    traja.color_dark(df.x, ax)


def test_sans_serif():
    traja.plotting.sans_serif()


def test_plot_flow():
    traja.plot_flow(df)


def test_plot_contour():
    fig = traja.plot_contour(df)


def test_plot_surface():
    fig = traja.plot_surface(df)


def test_plot_stream():
    fig = traja.plot_stream(df)


def test_trip_grid():
    traja.plotting.trip_grid(df)


def test_label_axes():
    df.traja.plot()
    ax = plt.gca()
    traja.plotting._label_axes(df, ax)


def test_plot_actogram():
    df = traja.generate(n=10)
    index = pd.DatetimeIndex(range(11))
    df.index = index
    activity = traja.calc_displacement(df)
    activity.name = "activity"
    traja.plotting.plot_actogram(df.x)


def test_plot_xy():
    traja.plotting.plot_xy(df)


def test_polar_bar():
    traja.plotting.polar_bar(df)


def test_find_runs():
    actual = traja.find_runs(df.x)
    expected = (
        np.array(
            [
                0.0,
                1.289_486_85,
                2.364_976_69,
                2.363_518_7,
                0.540_423_65,
                -1.308_330_34,
                -3.375_043_98,
                -5.424_061_97,
                -7.232_308_6,
                -9.174_619_43,
                -10.735_451_21,
            ]
        ),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    )
    for i in range(len(actual)):
        npt.assert_allclose(actual[i], expected[i])


def test_plot():
    fig = traja.plotting.plot(df)
    ax = fig.axes[1]
    coll = ax.collections[0]
    path = coll.get_paths()[0]
    npt.assert_allclose(
        path._vertices,
        np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 0.039_062_5],
                [0.0, 0.039_062_5],
                [0.0, 0.0],
            ]
        ),
    )
