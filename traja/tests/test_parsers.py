import os

import numpy as np
import pandas as pd

import traja


df = traja.generate(n=20)


def test_from_df():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4]})
    trj = traja.parsers.from_df(df)
    np.testing.assert_allclose(df, trj)
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
