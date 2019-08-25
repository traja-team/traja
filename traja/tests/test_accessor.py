import shapely
import pandas as pd

import traja

df = traja.generate(n=20)


def test_center():
    xy = df.traja.center


def test_night():
    df["time"] = pd.DatetimeIndex(range(20))
    df.traja.night()


def test_between():
    df["time"] = pd.DatetimeIndex(range(20))
    df.traja.between("8:00", "10:00")


def test_day():
    df["time"] = pd.DatetimeIndex(range(20))
    df.traja.day()


def test_xy():
    xy = df.traja.xy
    assert xy.shape == (20, 2)


def test_calc_derivatives():
    df.traja.calc_derivatives()


def test_get_derivatives():
    df.traja.get_derivatives()


def test_speed_intervals():
    si = df.traja.speed_intervals(faster_than=100)
    assert isinstance(si, traja.TrajaDataFrame)


def test_to_shapely():
    shape = df.traja.to_shapely()
    assert isinstance(shape, shapely.geometry.linestring.LineString)


def test_calc_displacement():
    disp = df.traja.calc_displacement()
    assert isinstance(disp, pd.Series)


def test_calc_angle():
    angle = df.traja.calc_angle()
    assert isinstance(angle, pd.Series)


def test_scale():
    df_copy = df.copy()
    df_copy.traja.scale(0.1)
    assert isinstance(df_copy, traja.TrajaDataFrame)


def test_rediscretize(R=0.1):
    df_copy = df.copy()
    r_df = df_copy.traja.rediscretize(R)
    assert isinstance(r_df, traja.TrajaDataFrame)
    assert r_df.shape == (382, 2)


def test_calc_heading():
    heading = df.traja.calc_heading()
    assert isinstance(heading, pd.Series)


def test_calc_turn_angle():
    turn_angle = df.traja.calc_turn_angle()
    assert isinstance(turn_angle, pd.Series)
