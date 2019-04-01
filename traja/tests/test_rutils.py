import numpy as np
import numpy.testing as npt
from pandas.util.testing import assert_frame_equal

import traja
import traja.rutils

df = traja.generate()


def test_import_adehabitat():
    from traja.rutils import import_adehabitat

    adehabitat = import_adehabitat()


def test_import_trajr():
    from traja.rutils import import_trajr

    trajr = import_trajr()


def test_plot_ltraj():
    ...


def test_to_trajr():
    trjr = traja.rutils.to_trajr(df)
    vals = trjr[0]
    actual = vals.r_repr()
    expected = (
        "c(0, 1.34184903653994, 2.36458915140914, 2.36329148545331, 0.54325141464563\n)"
    )
    assert actual == expected


def test_to_ltraj():
    ltraj = traja.rutils.to_ltraj(df)
    vals = ltraj[0][0]
    actual = vals.r_repr()[:20]
    expected = "c(0, 1.34184903653994, 2.36458915140914,"
    assert actual == expected
