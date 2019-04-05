import shapely

import numpy as np
import pandas as pd


import traja

df = traja.generate(n=20)


def test_center():
    xy = df.traja.center


def test_night():
    df["time"] = pd.DatetimeIndex(range(21))
    df.traja.night()


def test_between():
    df["time"] = pd.DatetimeIndex(range(21))
    df.traja.between("8:00", "10:00")


def test_day():
    df["time"] = pd.DatetimeIndex(range(21))
    df.traja.day()


def test_xy():
    xy = df.traja.xy
    assert xy.shape == (21, 2)


def test_calc_derivatives():
    df.traja.calc_derivatives()


def test_get_derivatives():
    df.traja.get_derivatives()


def test_speed_intervals():
    si = df.traja.speed_intervals
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
    df.traja.scale(0.1)
    assert isinstance(df, traja.TrajaDataFrame)


def test_rediscretize(R=0.1):
    r_df = df.traja.rediscretize(R)
    assert isinstance(r_df, traja.TrajaDataFrame)
    assert r_df.shape == (40, 2)


def test_calc_heading():
    heading = df.traja.calc_heading()
    assert isinstance(heading, pd.Series)


def test_calc_turn_angle():
    turn_angle = df.traja.calc_turn_angle()
    assert isinstance(turn_angle, pd.Series)
