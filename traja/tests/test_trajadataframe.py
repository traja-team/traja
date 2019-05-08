import os
import shutil
import tempfile

import pandas as pd

import traja
from traja import TrajaDataFrame, read_file


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
        assert isinstance(self.df, TrajaDataFrame)

    def test_copy(self):
        df2 = self.df.copy()
        assert (df2, TrajaDataFrame)
        assert df2.xlim == self.df.xlim

    def test_dataframe_to_trajadataframe(self):
        df = pd.DataFrame(
            {"x": range(len(self.df)), "y": range(len(self.df))}, index=self.df.index
        )

        tf = TrajaDataFrame(df)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(tf, TrajaDataFrame)

    def test_construct_dataframe(self):
        df = traja.TrajaDataFrame(
            {"x": range(len(self.df)), "y": range(len(self.df))},
            index=self.df.index,
            xlim=(0, 2),
            ylim=(0, 2),
            spatial_units="m",
            title="Serious title",
            fps=2.0,
            time_units="s",
            id=42,
        )

        assert df.title == "Serious title"

        # Test 'merge'
        df2 = df.copy()
        assert df2.title == "Serious title"

        assert df._get_time_col() == None
        assert self.df._get_time_col() == "Time"

        # Modify metavar
        df.set("title", "New title")
        assert df.title == "New title"

        # Test __finalize__
        df_copy = df.copy()
        df2_copy = df2.copy()
        assert isinstance(df_copy, traja.TrajaDataFrame)
