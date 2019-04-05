import warnings

import numpy as np
import numpy.testing as npt

import traja
from traja import rutils

warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

df = traja.generate(n=20)


def test_import_adehabitat():
    traja.rutils.import_adehabitat()


def test_import_trajr():
    traja.rutils.import_trajr()


def test_to_trajr():
    trjr = traja.rutils.to_trajr(df)
    assert "x" in trjr
    assert "y" in trjr
    assert "time" in trjr
    assert "polar" in trjr
    assert "displacement" in trjr
    actual = trjr.x[:3].values
    expected = np.array([0.0, 0.946_646_34, 1.946_959_33])

    npt.assert_allclose(actual, expected)


def test_to_ltraj():
    ltraj = traja.rutils.to_ltraj(df)
    rdataframe = ltraj
    assert "x" in rdataframe
    assert "y" in rdataframe
    assert len(rdataframe) == 21
