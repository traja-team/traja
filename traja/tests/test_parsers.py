import os

import pandas as pd

import traja

from pandas.util.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)

df = traja.generate(n=20)


def test_from_df():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4]})
    trj = traja.parsers.from_df(df)
    assert_frame_equal(df, trj)
    assert isinstance(trj, traja.TrajaDataFrame)


def test_read_file():
    datapath = os.path.join(traja.__path__[0], "tests", "data", "3527.csv")
    trj = traja.parsers.read_file(datapath)
    assert isinstance(trj, traja.TrajaDataFrame)
    assert "Frame" in trj
    assert "Time" in trj
    assert "TrackId" in trj
    assert "x" in trj
    assert "y" in trj
    assert "ValueChanged" in trj
