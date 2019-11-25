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
            1.162_605_74 + 1.412_179_34j,
            1.861_836_8 + 2.727_243_73j,
            1.860_393_36 + 4.857_966_96j,
            -0.096_486_29 + 5.802_456_77j,
            -1.735_291_68 + 4.940_704_34j,
            -4.189_217_4 + 4.951_826_17j,
            -5.712_624_22 + 4.177_006j,
            -7.567_193_14 + 3.404_176_98j,
            -9.415_289_13 + 2.743_725_89j,
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
            1.829_180_85,
            3.302_165_14,
            5.202_009_84,
            5.803_258_93,
            5.236_582_53,
            6.486_148_69,
            7.076_825_18,
            8.297_640_2,
            9.806_921_08,
        ]
    )
    theta_expected = np.array(
        [
            0.0,
            0.882_026_17,
            0.971_788_83,
            1.205_067_81,
            1.587_423_32,
            1.908_560_74,
            2.272_960_35,
            2.510_239_91,
            2.718_855_22,
            2.858_033_49,
        ]
    )

    npt.assert_allclose(r_actual[:10], r_expected)
    npt.assert_allclose(theta_actual[:10], theta_expected)


@pytest.mark.parametrize("eqn1", [True])
def test_expected_sq_displacement(eqn1):
    df_copy = df.copy()
    disp = traja.expected_sq_displacement(df_copy, eqn1=eqn1)
    if eqn1:
        npt.assert_allclose(disp, 0.757_882_272_948_632_8)


def test_step_lengths():
    df_copy = df.copy()
    step_lengths = traja.step_lengths(df_copy)
    actual = step_lengths.to_numpy()[:5]
    expected = np.array(
        [np.nan, 1.829_180_85, 1.489_402_04, 2.130_723_72, 2.172_887_24]
    )
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
                    [0.014_535_17, 0.041_638_09, 0.0],
                    [1.104_465_06, 1.245_626_99, 0.02],
                    [1.949_047_82, 2.977_072_25, 0.04],
                    [1.557_970_03, 4.739_519_81, 0.06],
                    [0.195_517, 5.519_674_6, 0.08],
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
                50.536_377_13,
                62.000_036_72,
                89.961_185_41,
                25.764_324_08,
                27.737_271_33,
                0.259_677_63,
                26.958_350_61,
                22.622_286,
                19.665_283_71,
                31.428_064_33,
                35.554_608_67,
                77.216_475_78,
                80.981_399_37,
                77.495_666_91,
                64.779_921_95,
                55.220_856_61,
                12.418_644_03,
                18.295_995_36,
                9.327_266_35,
            ]
        )
    elif lag == 2:
        expected = np.array(
            [
                np.nan,
                np.nan,
                55.679_398_79,
                78.552_154_19,
                57.510_652_27,
                1.318_153_96,
                11.741_160_75,
                10.869_226_84,
                24.615_298_57,
                21.161_131_62,
                26.022_239_16,
                33.485_645_28,
                55.060_685_9,
                88.237_494_22,
                79.351_771_4,
                71.545_102_77,
                59.557_726_58,
                33.248_128_63,
                15.505_016_09,
                13.817_221_74,
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
        -13.699_062_135_959_585,
        -10.144_216_927_960_029,
        1.861_836_800_674_031_3,
        5.802_456_768_595_229,
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
    expected = -2.707_106_781_186_548_3
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
    npt.assert_allclose(actual, 3.95)

    actual = grid_indices[:10].to_numpy()
    expected = np.array(
        [[8, 6], [9, 7], [9, 8], [9, 9], [8, 9], [7, 9], [6, 9], [5, 8], [3, 8], [2, 8]]
    )
    npt.assert_equal(actual, expected)


def test_generate():
    df = traja.generate(n=20)
    actual = df.traja.xy[:3]
    expected = np.array(
        [[0.0, 0.0], [1.162_605_74, 1.412_179_34], [1.861_836_8, 2.727_243_73]]
    )
    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_rotate():
    df_copy = df.copy()
    actual = traja.trajectory.rotate(df_copy, 10).traja.xy[:10]
    expected = np.array(
        [
            [18.646_466_67, 10.430_808_03],
            [16.902_701_92, 9.878_370_62],
            [15.600_574_26, 9.155_333_99],
            [14.442_626_99, 7.366_719_52],
            [15.570_766_59, 5.509_641_17],
            [17.414_653_05, 5.341_168_38],
            [19.467_621_74, 3.996_848_97],
            [21.167_387_56, 3.818_213_04],
            [23.143_938_84, 3.457_747_23],
            [25.053_922_91, 3.006_509_7],
        ]
    )
    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_rediscretize_points():
    df_copy = df.copy()
    actual = traja.rediscretize_points(df_copy, R=0.1)[:10].to_numpy()
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
            1.829_180_85,
            1.489_402_04,
            2.130_723_72,
            2.172_887_24,
            1.851_567,
            2.453_950_92,
            1.709_126_87,
            2.009_151_7,
            1.962_563_23,
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
            [1.829_180_85, 0.02],
            [1.489_402_04, 0.04],
            [2.130_723_72, 0.06],
            [2.172_887_24, 0.08],
            [1.851_567, 0.1],
            [2.453_950_92, 0.12],
            [1.709_126_87, 0.14],
            [2.009_151_7, 0.16],
            [1.962_563_23, 0.18],
        ]
    )

    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_calc_heading():
    df_copy = df.copy()
    actual = traja.calc_heading(df_copy)[:10].values
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

    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_get_derivatives():
    df_copy = df.copy()
    actual = traja.get_derivatives(df_copy)[:10].to_numpy()
    expected = np.array(
        [
            [np.nan, 0.000_000_00e00, np.nan, np.nan, np.nan, np.nan],
            [
                1.829_180_85e00,
                2.000_000_00e-02,
                9.145_904_26e01,
                2.000_000_00e-02,
                np.nan,
                np.nan,
            ],
            [
                1.489_402_04e00,
                4.000_000_00e-02,
                7.447_010_18e01,
                4.000_000_00e-02,
                -8.494_470_38e02,
                4.000_000_00e-02,
            ],
            [
                2.130_723_72e00,
                6.000_000_00e-02,
                1.065_361_86e02,
                6.000_000_00e-02,
                1.603_304_21e03,
                6.000_000_00e-02,
            ],
            [
                2.172_887_24e00,
                8.000_000_00e-02,
                1.086_443_62e02,
                8.000_000_00e-02,
                1.054_088_02e02,
                8.000_000_00e-02,
            ],
            [
                1.851_567_00e00,
                1.000_000_00e-01,
                9.257_834_98e01,
                1.000_000_00e-01,
                -8.033_006_10e02,
                1.000_000_00e-01,
            ],
            [
                2.453_950_92e00,
                1.200_000_00e-01,
                1.226_975_46e02,
                1.200_000_00e-01,
                1.505_959_82e03,
                1.200_000_00e-01,
            ],
            [
                1.709_126_87e00,
                1.400_000_00e-01,
                8.545_634_33e01,
                1.400_000_00e-01,
                -1.862_060_15e03,
                1.400_000_00e-01,
            ],
            [
                2.009_151_70e00,
                1.600_000_00e-01,
                1.004_575_85e02,
                1.600_000_00e-01,
                7.500_620_96e02,
                1.600_000_00e-01,
            ],
            [
                1.962_563_23e00,
                1.800_000_00e-01,
                9.812_816_15e01,
                1.800_000_00e-01,
                -1.164_711_84e02,
                1.800_000_00e-01,
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
                -13.699_062_14,
                -11.970_073_37,
                -10.241_084_59,
                -8.512_095_82,
                -6.783_107_05,
                -5.054_118_28,
                -3.325_129_51,
                -1.596_140_74,
                0.132_848_03,
                1.861_836_8,
            ],
            [
                -13.699_062_14,
                -11.970_073_37,
                -10.241_084_59,
                -8.512_095_82,
                -6.783_107_05,
                -5.054_118_28,
                -3.325_129_51,
                -1.596_140_74,
                0.132_848_03,
                1.861_836_8,
            ],
            [
                -13.699_062_14,
                -11.970_073_37,
                -10.241_084_59,
                -8.512_095_82,
                -6.783_107_05,
                -5.054_118_28,
                -3.325_129_51,
                -1.596_140_74,
                0.132_848_03,
                1.861_836_8,
            ],
            [
                -13.699_062_14,
                -11.970_073_37,
                -10.241_084_59,
                -8.512_095_82,
                -6.783_107_05,
                -5.054_118_28,
                -3.325_129_51,
                -1.596_140_74,
                0.132_848_03,
                1.861_836_8,
            ],
            [
                -13.699_062_14,
                -11.970_073_37,
                -10.241_084_59,
                -8.512_095_82,
                -6.783_107_05,
                -5.054_118_28,
                -3.325_129_51,
                -1.596_140_74,
                0.132_848_03,
                1.861_836_8,
            ],
            [
                -13.699_062_14,
                -11.970_073_37,
                -10.241_084_59,
                -8.512_095_82,
                -6.783_107_05,
                -5.054_118_28,
                -3.325_129_51,
                -1.596_140_74,
                0.132_848_03,
                1.861_836_8,
            ],
            [
                -13.699_062_14,
                -11.970_073_37,
                -10.241_084_59,
                -8.512_095_82,
                -6.783_107_05,
                -5.054_118_28,
                -3.325_129_51,
                -1.596_140_74,
                0.132_848_03,
                1.861_836_8,
            ],
            [
                -13.699_062_14,
                -11.970_073_37,
                -10.241_084_59,
                -8.512_095_82,
                -6.783_107_05,
                -5.054_118_28,
                -3.325_129_51,
                -1.596_140_74,
                0.132_848_03,
                1.861_836_8,
            ],
            [
                -13.699_062_14,
                -11.970_073_37,
                -10.241_084_59,
                -8.512_095_82,
                -6.783_107_05,
                -5.054_118_28,
                -3.325_129_51,
                -1.596_140_74,
                0.132_848_03,
                1.861_836_8,
            ],
            [
                -13.699_062_14,
                -11.970_073_37,
                -10.241_084_59,
                -8.512_095_82,
                -6.783_107_05,
                -5.054_118_28,
                -3.325_129_51,
                -1.596_140_74,
                0.132_848_03,
                1.861_836_8,
            ],
        ]
    )

    npt.assert_allclose(actual, expected, rtol=1e-1)


def test_from_xy():
    df_copy = df.copy()
    expected = traja.from_xy(df_copy.traja.xy).values
    actual = df_copy.traja.xy
    npt.assert_allclose(expected, actual)
