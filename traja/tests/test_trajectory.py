import numpy as np
import numpy.testing as npt
from pandas.util.testing import assert_series_equal

import traja

df = traja.generate(n=20)


def test_polar_to_z():
    polar = traja.cartesian_to_polar(df.traja.xy)
    z = traja.polar_to_z(*polar)
    z_expected = np.array(
        [
            0.0 + 0.0j,
            0.946_646_34 + 1.149_860_48j,
            1.946_959_33 + 3.031_178_5j,
            1.945_487_32 + 5.204_065_25j,
            0.277_985_34 + 6.008_886_65j,
            -1.893_984_67 + 4.866_773_96j,
            -3.603_093_98 + 4.874_520_09j,
            -5.393_923_83 + 3.963_685_85j,
            -7.205_488_76 + 3.208_777_29j,
            -9.377_513_86 + 2.432_564_17j,
        ]
    )

    npt.assert_allclose(z[:10], z_expected)


def test_cartesian_to_polar():
    xy = df.traja.xy

    r_actual, theta_actual = traja.cartesian_to_polar(xy)
    r_expected = np.array(
        [
            0.0,
            1.489_402_04,
            3.602_595_42,
            5.555_827_21,
            6.015_313_34,
            5.222_323_88,
            6.061_619_61,
            6.693_670_13,
            7.887_668_86,
            9.687_885_98,
        ]
    )

    theta_expected = np.array(
        [
            0.0,
            0.882_026_17,
            0.999_845_38,
            1.213_043_1,
            1.524_566_92,
            1.941_928_61,
            2.207_329_46,
            2.507_862_96,
            2.722_634_07,
            2.887_782_98,
        ]
    )

    npt.assert_allclose(r_actual[:10], r_expected)
    npt.assert_allclose(theta_actual[:10], theta_expected)


def test_expected_sq_displacement():
    ...


def test_traj_from_coords():
    coords = df.traja.xy
    trj = traja.traj_from_coords(coords, fps=50)
    assert "dt" in trj
    assert_series_equal(trj.x, df.x)
    assert_series_equal(trj.y, df.y)
    assert_series_equal(trj.time, df.time)


def test_distance():
    rotated = traja.trajectory.rotate(df, 10).traja.xy[:10]
    distance = traja.distance(rotated, df.traja.xy)
    npt.assert_almost_equal(distance, 523.103_701_021_348_1)


def test_to_shapely():
    actual = traja.to_shapely(df).bounds
    expected = (
        -13.446_470_734_189_983,
        -11.325_980_877_259_793,
        1.946_959_328_925_418_1,
        6.008_886_650_205_287,
    )
    npt.assert_allclose(actual, expected)


def test_transition_matrix():
    ...


def test_calculate_flow_angles():
    grid_indices = traja.grid_coordinates(df)
    U, V = traja.calculate_flow_angles(grid_indices.values)
    expected = -0.707_106_781_186_548_1
    npt.assert_approx_equal(U.sum(), expected)


def test_transitions():
    ...


def test_grid_coordinates():
    grid_indices = traja.trajectory.grid_coordinates(df)
    assert "xbin" in grid_indices
    assert "ybin" in grid_indices
    npt.assert_allclose(grid_indices.xbin.mean(), 4.761_904_761_904_762)

    actual = grid_indices[:10].to_numpy()
    expected = np.array(
        [
            [8, 7],
            [9, 8],
            [10, 9],
            [9, 10],
            [9, 11],
            [7, 10],
            [6, 10],
            [5, 9],
            [4, 9],
            [3, 8],
        ]
    )
    npt.assert_equal(actual, expected)


def test_generate():
    df = traja.generate(n=20)
    actual = df.traja.xy[:3]
    expected = np.array(
        [[0.0, 0.0], [0.946_646_34, 1.149_860_48], [1.946_959_33, 3.031_178_5]]
    )
    npt.assert_allclose(actual, expected)


def test_rotate():
    actual = traja.trajectory.rotate(df, 10).traja.xy[:10]

    expected = np.array(
        [
            [18.870_076_43, 11.752_855_2],
            [17.450_224_06, 11.303_035_6],
            [15.587_413_19, 10.268_666_6],
            [14.406_552_05, 8.444_658_4],
            [15.367_865_65, 6.862_199_39],
            [17.811_637_26, 6.638_916_09],
            [19.241_488_17, 5.702_624_99],
            [21.239_635_57, 5.492_630_82],
            [23.170_354_32, 5.140_523_54],
            [25.415_115_06, 4.610_194_36],
        ]
    )
    npt.assert_allclose(actual, expected)


def test_rediscretize_points():
    actual = traja.rediscretize_points(df, R=0.1)[:10].to_numpy()
    expected = np.array(
        [
            [0.0, 0.0],
            [0.063_558_82, 0.077_202_83],
            [0.127_117_64, 0.154_405_65],
            [0.190_676_46, 0.231_608_48],
            [0.254_235_27, 0.308_811_31],
            [0.317_794_09, 0.386_014_14],
            [0.381_352_91, 0.463_216_96],
            [0.444_911_73, 0.540_419_79],
            [0.508_470_55, 0.617_622_62],
            [0.572_029_37, 0.694_825_45],
        ]
    )
    npt.assert_allclose(actual, expected)


def test_calc_turn_angle():
    actual = traja.trajectory.calc_turn_angle(df).values[:10]
    npt.assert_allclose(
        actual,
        np.array(
            [
                np.nan,
                np.nan,
                11.463_659_59,
                28.038_777_87,
                64.196_861_33,
                53.501_595_42,
                -27.996_948_96,
                27.218_028_24,
                -4.336_064_62,
                -2.957_002_29,
            ]
        ),
    )


def test_calc_angle():
    ...


def test_calc_displacement():
    displacement = traja.calc_displacement(df)
    actual = displacement.values[:10]
    expected = np.array(
        [
            np.nan,
            1.489_402_04,
            2.130_723_72,
            2.172_887_24,
            1.851_567,
            2.453_950_92,
            1.709_126_87,
            2.009_151_7,
            1.962_563_23,
            2.306_555_84,
        ]
    )
    npt.assert_allclose(actual, expected)


def test_calc_derivatives():
    derivs = traja.calc_derivatives(df)
    assert "displacement" in derivs
    assert "displacement_time" in derivs
    actual = derivs.to_numpy()[:10]
    expected = np.array(
        [
            [np.nan, 0.0],
            [1.489_402_04, 0.02],
            [2.130_723_72, 0.04],
            [2.172_887_24, 0.06],
            [1.851_567, 0.08],
            [2.453_950_92, 0.1],
            [1.709_126_87, 0.12],
            [2.009_151_7, 0.14],
            [1.962_563_23, 0.16],
            [2.306_555_84, 0.18],
        ]
    )
    npt.assert_allclose(actual, expected)


def test_calc_heading():
    actual = traja.calc_heading(df)[:10].values
    expected = np.array(
        [
            np.nan,
            50.536_377_13,
            62.000_036_72,
            90.038_814_59,
            154.235_675_92,
            -152.262_728_67,
            179.740_322_37,
            -153.041_649_39,
            -157.377_714,
            -160.334_716_29,
        ]
    )
    npt.assert_allclose(actual, expected)


def test_get_derivatives():
    actual = traja.get_derivatives(df)[:10].to_numpy()
    expected = np.array(
        [
            [np.nan, 0.000_000_00e00, np.nan, np.nan, np.nan, np.nan],
            [
                1.489_402_04e00,
                2.000_000_00e-02,
                7.447_010_18e01,
                2.000_000_00e-02,
                np.nan,
                np.nan,
            ],
            [
                2.130_723_72e00,
                4.000_000_00e-02,
                1.065_361_86e02,
                4.000_000_00e-02,
                1.603_304_21e03,
                4.000_000_00e-02,
            ],
            [
                2.172_887_24e00,
                6.000_000_00e-02,
                1.086_443_62e02,
                6.000_000_00e-02,
                1.054_088_02e02,
                6.000_000_00e-02,
            ],
            [
                1.851_567_00e00,
                8.000_000_00e-02,
                9.257_834_98e01,
                8.000_000_00e-02,
                -8.033_006_10e02,
                8.000_000_00e-02,
            ],
            [
                2.453_950_92e00,
                1.000_000_00e-01,
                1.226_975_46e02,
                1.000_000_00e-01,
                1.505_959_82e03,
                1.000_000_00e-01,
            ],
            [
                1.709_126_87e00,
                1.200_000_00e-01,
                8.545_634_33e01,
                1.200_000_00e-01,
                -1.862_060_15e03,
                1.200_000_00e-01,
            ],
            [
                2.009_151_70e00,
                1.400_000_00e-01,
                1.004_575_85e02,
                1.400_000_00e-01,
                7.500_620_96e02,
                1.400_000_00e-01,
            ],
            [
                1.962_563_23e00,
                1.600_000_00e-01,
                9.812_816_15e01,
                1.600_000_00e-01,
                -1.164_711_84e02,
                1.600_000_00e-01,
            ],
            [
                2.306_555_84e00,
                1.800_000_00e-01,
                1.153_277_92e02,
                1.800_000_00e-01,
                8.599_815_32e02,
                1.800_000_00e-01,
            ],
        ]
    )
    npt.assert_allclose(actual, expected)


def test_coords_to_flow():
    grid_flow = traja.coords_to_flow(df)[:10]
    actual = grid_flow[0]
    expected = np.array(
        [
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
            [
                -13.446_470_73,
                -11.736_089_62,
                -10.025_708_5,
                -8.315_327_38,
                -6.604_946_26,
                -4.894_565_14,
                -3.184_184_03,
                -1.473_802_91,
                0.236_578_21,
                1.946_959_33,
            ],
        ]
    )
    npt.assert_allclose(actual, expected)


def test_from_xy():
    expected = traja.from_xy(df.traja.xy).values[:10]
    actual = np.array(
        [
            [0.0, 0.0],
            [0.946_646_34, 1.149_860_48],
            [1.946_959_33, 3.031_178_5],
            [1.945_487_32, 5.204_065_25],
            [0.277_985_34, 6.008_886_65],
            [-1.893_984_67, 4.866_773_96],
            [-3.603_093_98, 4.874_520_09],
            [-5.393_923_83, 3.963_685_85],
            [-7.205_488_76, 3.208_777_29],
            [-9.377_513_86, 2.432_564_17],
        ]
    )
    npt.assert_allclose(expected, actual)
