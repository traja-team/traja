import warnings

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
    vals = trjr[0]
    actual = vals.r_repr()
    expected = "c(0, 0.946646340454933, 1.94695932892542, 1.94548732303614, 0.277985344978653, \n"
    assert expected in actual


def test_to_ltraj():
    ltraj = traja.rutils.to_ltraj(df)
    rdataframe = ltraj[0]
    assert "x" in rdataframe
    assert "y" in rdataframe
    assert len(rdataframe) == 21
