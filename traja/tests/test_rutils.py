import warnings

import traja
import traja.rutils

warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

df = traja.generate(n=20)


def test_import_adehabitat():
    from traja.rutils import import_adehabitat

    adehabitat = import_adehabitat()


def test_import_trajr():
    from traja.rutils import import_trajr

    trajr = import_trajr()


def test_to_trajr():
    trjr = traja.rutils.to_trajr(df)
    vals = trjr[0]
    actual = vals.r_repr()
    expected = "c(0, 0.946646340454933, 1.94695932892542, 1.94548732303614, 0.277985344978653, \n"
    assert expected in actual


def test_to_ltraj():
    ltraj = traja.rutils.to_ltraj(df)
    vals = ltraj[0][0]
    actual = vals.r_repr()[:24]
    expected = "c(0, 0.946646340454933, "
    assert actual == expected
