import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import traja
from traja import TrajaDataFrame, read_file

import pytest
from pandas.util.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)


class TestDataFrame:
    def setup_method(self):
        dirname = os.path.dirname(traja.__file__)
        data_filename = os.path.join(dirname, "tests/data/3527.csv")
        df = read_file(data_filename)
        self.df = read_file(data_filename, xlim=(df.x.min(), df.x.max()))
        self.tempdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tempdir)

    def test_df_init(self):
        assert type(self.df) is TrajaDataFrame

    def test_copy(self):
        df2 = self.df.copy()
        assert type(df2) is TrajaDataFrame
        assert df2.xlim == self.df.xlim

    def test_dataframe_to_trajadataframe(self):
        df = pd.DataFrame(
            {"x": range(len(self.df)), "y": range(len(self.df))}, index=self.df.index
        )

        tf = TrajaDataFrame(df)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(tf, TrajaDataFrame)
