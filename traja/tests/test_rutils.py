import warnings

import numpy as np
import numpy.testing as npt
from rpy2 import rinterface
from rpy2.rinterface import RRuntimeWarning

import traja
from traja import rutils


warnings.filterwarnings("ignore", category=RRuntimeWarning, module="rpy2")
warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")


df = traja.generate(n=20)


def test_import_adehabitat():
    rutils.import_adehabitat(suppress_messages=True)


def test_import_trajr():
    rutils.import_trajr(suppress_messages=True)


def test_to_trajr():
    trjr = rutils.to_trajr(df)
    assert "x" in trjr
    assert "y" in trjr
    assert "time" in trjr
    assert "polar" in trjr
    assert "displacement" in trjr
    actual = trjr.x[:3].values
    expected = np.array([0.0, 1.162_605_74, 1.861_836_8])

    npt.assert_allclose(actual, expected)


def test_to_ltraj():
    ltraj = rutils.to_ltraj(df)
    rdataframe = ltraj
    assert "x" in rdataframe
    assert "y" in rdataframe
    assert len(rdataframe) == 20
