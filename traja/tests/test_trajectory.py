import numpy as np
import numpy.testing as npt
import pytest
from pandas.util.testing import assert_series_equal

import traja

df = traja.generate(n=20)


def test_polar_to_z():
    df_copy = df.copy()
    polar = traja.cartesian_to_polar(df_copy.traja.xy)
    z = traja.polar_to_z(*polar)
    z_actual = z[:10]
    z_expected = np.array(
        [
            0.0 + 0.0j,
            1.16260574 + 1.41217934j,
            1.8618368 + 2.72724373j,
            1.86039336 + 4.85796696j,
            -0.09648629 + 5.80245677j,
            -1.73529168 + 4.94070434j,
            -4.1892174 + 4.95182617j,
            -5.71262422 + 4.177006j,
            -7.56719314 + 3.40417698j,
            -9.41528913 + 2.74372589j,
        ]
    )

    npt.assert_allclose(z_actual, z_expected)


def test_cartesian_to_polar():
    df_copy = df.copy()
    xy = df_copy.traja.xy

    r_actual, theta_actual = traja.cartesian_to_polar(xy)
    r_expected = np.array(
        [
            0.0,
            1.82918085,
            3.30216514,
            5.20200984,
            5.80325893,
            5.23658253,
            6.48614869,
            7.07682518,
            8.2976402,
            9.80692108,
        ]
    )
    theta_expected = np.array(
        [
            0.0,
            0.88202617,
            0.97178883,
            1.20506781,
            1.58742332,
            1.90856074,
            2.27296035,
            2.51023991,
            2.71885522,
            2.85803349,
        ]
    )

    npt.assert_allclose(r_actual[:10], r_expected)
    npt.assert_allclose(theta_actual[:10], theta_expected)


@pytest.mark.parametrize("eqn1", [True])
def test_expected_sq_displacement(eqn1):
    df_copy = df.copy()
    disp = traja.expected_sq_displacement(df_copy, eqn1=eqn1)
    if eqn1:
        npt.assert_allclose(disp, 0.7578822729486328)


def test_step_lengths():
    df_copy = df.copy()
    step_lengths = traja.step_lengths(df_copy)
    actual = step_lengths.to_numpy()[:5]
    expected = np.array([np.nan, 1.82918085, 1.48940204, 2.13072372, 2.17288724])
    npt.assert_allclose(actual, expected)
    assert len(step_lengths == len(df_copy))


@pytest.mark.parametrize("w", [None, 6])
def test_smooth_sg(w):
    df_copy = df.copy()
    if w == 6:
        with pytest.raises(Exception):
            _ = traja.trajectory.smooth_sg(df_copy, w=w)
    else:
        trj = traja.trajectory.smooth_sg(df_copy, w=w)
        actual = trj.to_numpy()[:5]
        if w is None:  # 5 with default settings
            expected = np.array(
                [
                    [0.01453517, 0.04163809, 0.0],
                    [1.10446506, 1.24562699, 0.02],
                    [1.94904782, 2.97707225, 0.04],
                    [1.55797003, 4.73951981, 0.06],
                    [0.195517, 5.5196746, 0.08],
                ]
            )
            npt.assert_allclose(actual, expected)
        else:
            raise Exception(f"Not tested w=={w}")
        assert trj.shape == df_copy.shape


@pytest.mark.parametrize("lag", [1, 2])
def test_angles(lag):
    df_copy = df.copy()
    angles = traja.angles(df_copy, lag=lag)
    actual = angles.to_numpy()
    if lag == 1:
        expected = np.array(
            [
                np.nan,
                50.53637713,
                62.00003672,
                89.96118541,
                25.76432408,
                27.73727133,
                0.25967763,
                26.95835061,
                22.622286,
                19.66528371,
                31.42806433,
                35.55460867,
                77.21647578,
                80.98139937,
                77.49566691,
                64.77992195,
                55.22085661,
                12.41864403,
                18.29599536,
                9.32726635,
            ]
        )
    elif lag == 2:
        expected = np.array(
            [
                np.nan,
                np.nan,
                55.67939879,
                78.55215419,
                57.51065227,
                1.31815396,
                11.74116075,
                10.86922684,
                24.61529857,
                21.16113162,
                26.02223916,
                33.48564528,
                55.0606859,
                88.23749422,
                79.3517714,
                71.54510277,
                59.55772658,
                33.24812863,
                15.50501609,
                13.81722174,
            ]
        )

    npt.assert_allclose(actual, expected)


def test_traj_from_coords():
    df_copy = df.copy()
    coords = df_copy.traja.xy
    trj = traja.traj_from_coords(coords, fps=50)
    assert "dt" in trj
    assert_series_equal(trj.x, df_copy.x)
    assert_series_equal(trj.y, df_copy.y)
    assert_series_equal(trj.time, df_copy.time)


@pytest.mark.parametrize("method", ["dtw", "hausdorff"])
def test_distance(method):
    df_copy = df.copy()
    rotated = traja.trajectory.rotate(df_copy, 10).traja.xy[:10]
    distance = traja.distance_between(rotated, df_copy.traja.xy, method=method)


@pytest.mark.parametrize("ndarray_type", [True, False])
def test_grid_coords1D(ndarray_type):
    df_copy = df.copy()
    xlim, ylim = traja.trajectory._get_xylim(df_copy)
    bins = traja.trajectory._bins_to_tuple(df_copy, None)
    grid_indices = traja.grid_coordinates(df_copy, bins=bins, xlim=xlim, ylim=ylim)
    if ndarray_type:
        grid_indices = grid_indices.values
    grid_indices1D = traja._grid_coords1D(grid_indices)
    assert isinstance(grid_indices1D, np.ndarray)


def test_to_shapely():
    df_copy = df.copy()
    actual = traja.to_shapely(df_copy).bounds
    expected = (
        -13.699062135959585,
        -10.144216927960029,
        1.8618368006740313,
        5.802456768595229,
    )
    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_transition_matrix():
    df_copy = df.copy()
    grid_indices = traja.grid_coordinates(df_copy)
    assert grid_indices.shape[1] == 2
    grid_indices1D = traja._grid_coords1D(grid_indices)
    transitions_matrix = traja.transition_matrix(grid_indices1D)


def test_calculate_flow_angles():
    df_copy = df.copy()
    grid_indices = traja.grid_coordinates(df_copy)
    U, V = traja.calculate_flow_angles(grid_indices.values)
    actual = U.sum()
    expected = -2.121320343559644
    npt.assert_allclose(actual, expected)


def test_resample_time():
    df_copy = df.copy()
    trj = traja.resample_time(df_copy, "3s")
    assert isinstance(trj, traja.TrajaDataFrame)


def test_transitions():
    df_copy = df.copy()
    transitions = traja.transitions(df_copy)
    assert isinstance(transitions, np.ndarray)

    # Check when bins set
    bins = traja._bins_to_tuple(df_copy, bins=None)
    xmin = df_copy.x.min()
    xmax = df_copy.x.max()
    ymin = df_copy.y.min()
    ymax = df_copy.y.max()
    xbins = np.linspace(xmin, xmax, bins[0])
    ybins = np.linspace(ymin, ymax, bins[1])
    xbin = np.digitize(df_copy.x, xbins)
    ybin = np.digitize(df_copy.y, ybins)

    df_copy.set("xbin", xbin)
    df_copy.set("ybin", ybin)
    transitions = traja.transitions(df_copy)
    assert isinstance(transitions, np.ndarray)


def test_grid_coordinates():
    df_copy = df.copy()
    grid_indices = traja.trajectory.grid_coordinates(df_copy)
    assert "xbin" in grid_indices
    assert "ybin" in grid_indices
    actual = grid_indices.xbin.mean()
    npt.assert_allclose(actual, 4.55)

    actual = grid_indices[:10].to_numpy()
    expected = np.array(
        [
            [8, 6],
            [9, 7],
            [10, 8],
            [9, 9],
            [8, 10],
            [7, 9],
            [6, 9],
            [5, 9],
            [4, 8],
            [3, 8],
        ]
    )
    npt.assert_equal(actual, expected)


def test_generate():
    df = traja.generate(n=20)
    actual = df.traja.xy[:3]
    expected = np.array([[0.0, 0.0], [1.16260574, 1.41217934], [1.8618368, 2.72724373]])
    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_rotate():
    df_copy = df.copy()
    actual = traja.trajectory.rotate(df_copy, 10).traja.xy[:10]
    expected = np.array(
        [
            [18.64646667, 10.43080803],
            [16.90270192, 9.87837062],
            [15.60057426, 9.15533399],
            [14.44262699, 7.36671952],
            [15.57076659, 5.50964117],
            [17.41465305, 5.34116838],
            [19.46762174, 3.99684897],
            [21.16738756, 3.81821304],
            [23.14393884, 3.45774723],
            [25.05392291, 3.0065097],
        ]
    )
    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_rediscretize_points():
    df_copy = df.copy()
    actual = traja.rediscretize_points(df_copy, R=0.1)[:10].to_numpy()
    expected = np.array(
        [
            [0.0, 0.0],
            [0.06355882, 0.07720283],
            [0.12711764, 0.15440565],
            [0.19067646, 0.23160848],
            [0.25423527, 0.30881131],
            [0.31779409, 0.38601414],
            [0.38135291, 0.46321696],
            [0.44491173, 0.54041979],
            [0.50847055, 0.61762262],
            [0.57202937, 0.69482545],
        ]
    )

    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_calc_turn_angle():
    df_copy = df.copy()
    actual = traja.trajectory.calc_turn_angle(df_copy).values[:10]
    npt.assert_allclose(
        actual,
        np.array(
            [
                np.nan,
                np.nan,
                11.46365959,
                28.03877787,
                64.19686133,
                53.50159542,
                -27.99694896,
                27.21802824,
                -4.33606462,
                -2.95700229,
            ]
        ),
        rtol=1e-1,
    )


def test_calc_angle():
    ...


def test_calc_displacement():
    df_copy = df.copy()
    displacement = traja.calc_displacement(df_copy)
    actual = displacement.values[:10]
    expected = np.array(
        [
            np.nan,
            1.82918085,
            1.48940204,
            2.13072372,
            2.17288724,
            1.851567,
            2.45395092,
            1.70912687,
            2.0091517,
            1.96256323,
        ]
    )

    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_calc_derivatives():
    df_copy = df.copy()
    derivs = traja.calc_derivatives(df_copy)
    assert "displacement" in derivs
    assert "displacement_time" in derivs
    actual = derivs.to_numpy()[:10]
    expected = np.array(
        [
            [np.nan, 0.0],
            [1.82918085, 0.02],
            [1.48940204, 0.04],
            [2.13072372, 0.06],
            [2.17288724, 0.08],
            [1.851567, 0.1],
            [2.45395092, 0.12],
            [1.70912687, 0.14],
            [2.0091517, 0.16],
            [1.96256323, 0.18],
        ]
    )

    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_calc_heading():
    df_copy = df.copy()
    actual = traja.calc_heading(df_copy)[:10].values
    expected = np.array(
        [
            np.nan,
            50.53637713,
            62.00003672,
            90.03881459,
            154.23567592,
            -152.26272867,
            179.74032237,
            -153.04164939,
            -157.377714,
            -160.33471629,
        ]
    )

    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_get_derivatives():
    df_copy = df.copy()
    actual = traja.get_derivatives(df_copy)[:10].to_numpy()
    expected = np.array(
        [
            [np.nan, 0.00000000e00, np.nan, np.nan, np.nan, np.nan],
            [
                1.82918085e00,
                2.00000000e-02,
                9.14590426e01,
                2.00000000e-02,
                np.nan,
                np.nan,
            ],
            [
                1.48940204e00,
                4.00000000e-02,
                7.44701018e01,
                4.00000000e-02,
                -8.49447038e02,
                4.00000000e-02,
            ],
            [
                2.13072372e00,
                6.00000000e-02,
                1.06536186e02,
                6.00000000e-02,
                1.60330421e03,
                6.00000000e-02,
            ],
            [
                2.17288724e00,
                8.00000000e-02,
                1.08644362e02,
                8.00000000e-02,
                1.05408802e02,
                8.00000000e-02,
            ],
            [
                1.85156700e00,
                1.00000000e-01,
                9.25783498e01,
                1.00000000e-01,
                -8.03300610e02,
                1.00000000e-01,
            ],
            [
                2.45395092e00,
                1.20000000e-01,
                1.22697546e02,
                1.20000000e-01,
                1.50595982e03,
                1.20000000e-01,
            ],
            [
                1.70912687e00,
                1.40000000e-01,
                8.54563433e01,
                1.40000000e-01,
                -1.86206015e03,
                1.40000000e-01,
            ],
            [
                2.00915170e00,
                1.60000000e-01,
                1.00457585e02,
                1.60000000e-01,
                7.50062096e02,
                1.60000000e-01,
            ],
            [
                1.96256323e00,
                1.80000000e-01,
                9.81281615e01,
                1.80000000e-01,
                -1.16471184e02,
                1.80000000e-01,
            ],
        ]
    )

    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_coords_to_flow():
    df_copy = df.copy()
    grid_flow = traja.coords_to_flow(df_copy)[:10]
    actual = grid_flow[0]
    expected = np.array(
        [
            [
                -13.69906214,
                -11.97007337,
                -10.24108459,
                -8.51209582,
                -6.78310705,
                -5.05411828,
                -3.32512951,
                -1.59614074,
                0.13284803,
                1.8618368,
            ],
            [
                -13.69906214,
                -11.97007337,
                -10.24108459,
                -8.51209582,
                -6.78310705,
                -5.05411828,
                -3.32512951,
                -1.59614074,
                0.13284803,
                1.8618368,
            ],
            [
                -13.69906214,
                -11.97007337,
                -10.24108459,
                -8.51209582,
                -6.78310705,
                -5.05411828,
                -3.32512951,
                -1.59614074,
                0.13284803,
                1.8618368,
            ],
            [
                -13.69906214,
                -11.97007337,
                -10.24108459,
                -8.51209582,
                -6.78310705,
                -5.05411828,
                -3.32512951,
                -1.59614074,
                0.13284803,
                1.8618368,
            ],
            [
                -13.69906214,
                -11.97007337,
                -10.24108459,
                -8.51209582,
                -6.78310705,
                -5.05411828,
                -3.32512951,
                -1.59614074,
                0.13284803,
                1.8618368,
            ],
            [
                -13.69906214,
                -11.97007337,
                -10.24108459,
                -8.51209582,
                -6.78310705,
                -5.05411828,
                -3.32512951,
                -1.59614074,
                0.13284803,
                1.8618368,
            ],
            [
                -13.69906214,
                -11.97007337,
                -10.24108459,
                -8.51209582,
                -6.78310705,
                -5.05411828,
                -3.32512951,
                -1.59614074,
                0.13284803,
                1.8618368,
            ],
            [
                -13.69906214,
                -11.97007337,
                -10.24108459,
                -8.51209582,
                -6.78310705,
                -5.05411828,
                -3.32512951,
                -1.59614074,
                0.13284803,
                1.8618368,
            ],
            [
                -13.69906214,
                -11.97007337,
                -10.24108459,
                -8.51209582,
                -6.78310705,
                -5.05411828,
                -3.32512951,
                -1.59614074,
                0.13284803,
                1.8618368,
            ],
            [
                -13.69906214,
                -11.97007337,
                -10.24108459,
                -8.51209582,
                -6.78310705,
                -5.05411828,
                -3.32512951,
                -1.59614074,
                0.13284803,
                1.8618368,
            ],
        ]
    )

    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_from_xy():
    df_copy = df.copy()
    expected = traja.from_xy(df_copy.traja.xy).values
    actual = df_copy.traja.xy
    npt.assert_allclose(expected, actual)
