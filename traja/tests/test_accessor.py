import pandas as pd
import shapely

import traja

df = traja.generate(n=20)


def test_center():
    xy = df.traja.center


def test_night():
    df["time"] = pd.DatetimeIndex(range(20))
    df.traja.night()


def test_between():
    df["time"] = pd.DatetimeIndex(range(20))
    df.traja.between("8:00", "10:00")


def test_day():
    df["time"] = pd.DatetimeIndex(range(20))
    df.traja.day()


def test_xy():
    xy = df.traja.xy
    assert xy.shape == (20, 2)


# def test_calc_derivatives():
#    df.traja.calc_derivatives()


# def test_get_derivatives():
#    df.traja.get_derivatives()


# def test_speed_intervals():
#    si = df.traja.speed_intervals(faster_than=100)
#    assert isinstance(si, traja.TrajaDataFrame)


def test_to_shapely():
    shape = df.traja.to_shapely()
    assert isinstance(shape, shapely.geometry.linestring.LineString)


def test_calc_displacement():
    disp = df.traja.calc_displacement()
    assert isinstance(disp, pd.Series)


def test_calc_angle():
    angle = df.traja.calc_angle()
    assert isinstance(angle, pd.Series)


def test_scale():
    df_copy = df.copy()
    df_copy.traja.scale(0.1)
    assert isinstance(df_copy, traja.TrajaDataFrame)


def test_rediscretize(R=0.1):
    df_copy = df.copy()
    r_df = df_copy.traja.rediscretize(R)
    assert isinstance(r_df, traja.TrajaDataFrame)
    assert r_df.shape == (382, 2)


def test_calc_heading():
    heading = df.traja.calc_heading()
    assert isinstance(heading, pd.Series)


def test_calc_turn_angle():
    turn_angle = df.traja.calc_turn_angle()
    assert isinstance(turn_angle, pd.Series)


# ============================================================================
# Deep Learning Features Tests
# ============================================================================


def test_extract_features():
    """Test feature extraction for ML."""
    features = df.traja.extract_features()
    assert isinstance(features, pd.DataFrame)
    assert 'displacement' in features.columns
    assert len(features) == len(df)


def test_to_tensor():
    """Test PyTorch tensor conversion."""
    result = df.traja.to_tensor()
    # Should return tensor if torch available, otherwise numpy array
    assert hasattr(result, 'shape')
    assert result.shape[0] == len(df)
    assert result.shape[1] == 2  # x, y coordinates


def test_to_tensor_with_z():
    """Test tensor conversion with z coordinate."""
    df_3d = df.copy()
    df_3d['z'] = pd.Series(range(len(df)), dtype=float)
    result = df_3d.traja.to_tensor()
    assert hasattr(result, 'shape')
    assert result.shape[1] == 3  # x, y, z coordinates


# Data Augmentation Tests


def test_augment_rotate():
    """Test rotation augmentation."""
    rotated = df.traja.augment_rotate(angle=90)
    assert isinstance(rotated, traja.TrajaDataFrame)
    assert len(rotated) == len(df)
    assert 'x' in rotated.columns and 'y' in rotated.columns


def test_augment_rotate_random():
    """Test rotation with random angle."""
    rotated = df.traja.augment_rotate()  # Random angle
    assert isinstance(rotated, traja.TrajaDataFrame)
    assert len(rotated) == len(df)


def test_augment_noise():
    """Test noise augmentation."""
    noisy = df.traja.augment_noise(sigma=0.1)
    assert isinstance(noisy, traja.TrajaDataFrame)
    assert len(noisy) == len(df)
    # Check that coordinates changed but not too much
    import numpy as np
    assert not np.allclose(df.x.values, noisy.x.values)


def test_augment_reverse():
    """Test time reversal augmentation."""
    reversed_df = df.traja.augment_reverse()
    assert isinstance(reversed_df, traja.TrajaDataFrame)
    assert len(reversed_df) == len(df)
    # First point should be last point of original
    import numpy as np
    assert np.allclose(reversed_df.iloc[0][['x', 'y']].values,
                      df.iloc[-1][['x', 'y']].values)


def test_augment_scale():
    """Test scaling augmentation."""
    scaled = df.traja.augment_scale(factor=2.0)
    assert isinstance(scaled, traja.TrajaDataFrame)
    assert len(scaled) == len(df)
    # Coordinates should be roughly 2x
    import numpy as np
    assert np.allclose(scaled.x.values, df.x.values * 2.0, rtol=0.01)


def test_augment_scale_random():
    """Test scaling with random factor."""
    scaled = df.traja.augment_scale()  # Random factor
    assert isinstance(scaled, traja.TrajaDataFrame)
    assert len(scaled) == len(df)


def test_augment_subsample():
    """Test subsampling augmentation."""
    subsampled = df.traja.augment_subsample(step=2)
    assert isinstance(subsampled, traja.TrajaDataFrame)
    assert len(subsampled) == len(df) // 2
    # Check indices match every 2nd point
    import numpy as np
    assert np.allclose(subsampled.iloc[0][['x', 'y']].values,
                      df.iloc[0][['x', 'y']].values)


# Sequence Processing Tests


def test_pad_trajectory_edge():
    """Test trajectory padding with edge mode."""
    df_short = df.iloc[:10].copy()
    padded = df_short.traja.pad_trajectory(target_length=20, mode='edge')
    assert isinstance(padded, traja.TrajaDataFrame)
    assert len(padded) == 20
    # Last padded values should match last original value
    import numpy as np
    assert np.allclose(padded.iloc[-1][['x', 'y']].values,
                      df_short.iloc[-1][['x', 'y']].values)


def test_pad_trajectory_constant():
    """Test trajectory padding with constant (zero) mode."""
    df_short = df.iloc[:10].copy()
    padded = df_short.traja.pad_trajectory(target_length=20, mode='constant')
    assert isinstance(padded, traja.TrajaDataFrame)
    assert len(padded) == 20


def test_pad_trajectory_linear():
    """Test trajectory padding with linear extrapolation."""
    df_short = df.iloc[:10].copy()
    padded = df_short.traja.pad_trajectory(target_length=20, mode='linear')
    assert isinstance(padded, traja.TrajaDataFrame)
    assert len(padded) == 20


def test_pad_trajectory_error():
    """Test that padding raises error if target < current length."""
    import pytest
    with pytest.raises(ValueError):
        df.traja.pad_trajectory(target_length=10)


def test_truncate_trajectory_end():
    """Test trajectory truncation from end."""
    truncated = df.traja.truncate_trajectory(target_length=10, mode='end')
    assert isinstance(truncated, traja.TrajaDataFrame)
    assert len(truncated) == 10
    # Should keep first 10 points
    import numpy as np
    assert np.allclose(truncated.iloc[0][['x', 'y']].values,
                      df.iloc[0][['x', 'y']].values)


def test_truncate_trajectory_start():
    """Test trajectory truncation from start."""
    truncated = df.traja.truncate_trajectory(target_length=10, mode='start')
    assert isinstance(truncated, traja.TrajaDataFrame)
    assert len(truncated) == 10
    # Should keep last 10 points
    import numpy as np
    assert np.allclose(truncated.iloc[0][['x', 'y']].values,
                      df.iloc[-10][['x', 'y']].values)


def test_truncate_trajectory_random():
    """Test trajectory truncation with random start."""
    truncated = df.traja.truncate_trajectory(target_length=10, mode='random')
    assert isinstance(truncated, traja.TrajaDataFrame)
    assert len(truncated) == 10


def test_truncate_trajectory_error():
    """Test that truncation raises error if target > current length."""
    import pytest
    with pytest.raises(ValueError):
        df.traja.truncate_trajectory(target_length=100)


def test_normalize_trajectory():
    """Test trajectory normalization."""
    normalized = df.traja.normalize_trajectory(scale=True, center=True)
    assert isinstance(normalized, traja.TrajaDataFrame)
    assert len(normalized) == len(df)
    # Check that mean is close to zero
    import numpy as np
    assert abs(normalized.x.mean()) < 1e-10
    assert abs(normalized.y.mean()) < 1e-10
    # Check that std is close to 1 (use looser tolerance for small sample sizes)
    assert abs(normalized.x.std() - 1.0) < 0.05
    assert abs(normalized.y.std() - 1.0) < 0.05


def test_normalize_trajectory_center_only():
    """Test trajectory normalization with centering only."""
    normalized = df.traja.normalize_trajectory(scale=False, center=True)
    assert isinstance(normalized, traja.TrajaDataFrame)
    import numpy as np
    assert abs(normalized.x.mean()) < 1e-10


def test_normalize_trajectory_scale_only():
    """Test trajectory normalization with scaling only."""
    normalized = df.traja.normalize_trajectory(scale=True, center=False)
    assert isinstance(normalized, traja.TrajaDataFrame)
    import numpy as np
    assert abs(normalized.x.std() - 1.0) < 0.05


# 3D Support Tests


def test_has_z():
    """Test _has_z() method."""
    df_2d = df.copy()
    df_3d = df.copy()
    df_3d['z'] = pd.Series(range(len(df)), dtype=float)

    assert not df_2d.traja._has_z()
    assert df_3d.traja._has_z()


def test_xyz_property():
    """Test xyz property for 3D trajectories."""
    df_3d = df.copy()
    df_3d['z'] = pd.Series(range(len(df)), dtype=float)

    xyz = df_3d.traja.xyz
    import numpy as np
    assert isinstance(xyz, np.ndarray)
    assert xyz.shape == (len(df), 3)


def test_center_3d():
    """Test center property for 3D trajectories."""
    df_3d = df.copy()
    df_3d['z'] = pd.Series(range(len(df)), dtype=float)

    center = df_3d.traja.center
    assert len(center) == 3  # x, y, z


def test_bounds_3d():
    """Test bounds property for 3D trajectories."""
    df_3d = df.copy()
    df_3d['z'] = pd.Series(range(len(df)), dtype=float)

    bounds = df_3d.traja.bounds
    assert len(bounds) == 3  # xlim, ylim, zlim
    assert all(len(lim) == 2 for lim in bounds)


# Export Methods Tests


def test_to_csv(tmp_path):
    """Test CSV export."""
    import os
    csv_path = tmp_path / "test_traj.csv"
    df.traja.to_csv(str(csv_path))
    assert os.path.exists(csv_path)
    # Read back and verify
    df_loaded = pd.read_csv(csv_path)
    assert len(df_loaded) == len(df)


def test_to_npy(tmp_path):
    """Test NumPy export."""
    import os
    import numpy as np
    npy_path = tmp_path / "test_traj.npy"
    df.traja.to_npy(str(npy_path))
    assert os.path.exists(npy_path)
    # Read back and verify
    data = np.load(npy_path)
    assert data.shape == (len(df), 2)


def test_to_npy_3d(tmp_path):
    """Test NumPy export with 3D data."""
    import os
    import numpy as np
    df_3d = df.copy()
    df_3d['z'] = pd.Series(range(len(df)), dtype=float)

    npy_path = tmp_path / "test_traj_3d.npy"
    df_3d.traja.to_npy(str(npy_path))
    assert os.path.exists(npy_path)
    # Read back and verify
    data = np.load(npy_path)
    assert data.shape == (len(df), 3)


def test_summary():
    """Test summary method."""
    summary = df.traja.summary()
    assert isinstance(summary, dict)
    assert 'n_points' in summary
    assert 'center' in summary
    assert 'bounds' in summary
    assert 'distance' in summary
    assert 'length' in summary
    assert 'dimensionality' in summary
    assert summary['n_points'] == len(df)
    assert summary['dimensionality'] == '2D'


def test_summary_3d():
    """Test summary method with 3D data."""
    df_3d = df.copy()
    df_3d['z'] = pd.Series(range(len(df)), dtype=float)

    summary = df_3d.traja.summary()
    assert summary['dimensionality'] == '3D'


def test_speed_property():
    """Test speed property."""
    speed = df.traja.speed
    assert isinstance(speed, pd.Series)
    assert len(speed) == len(df)


def test_displacement_property():
    """Test displacement property."""
    displacement = df.traja.displacement
    assert isinstance(displacement, pd.Series)
    assert len(displacement) == len(df)


# ============================================================================
# Visualization Features Tests
# ============================================================================


def test_plot_heatmap():
    """Test heatmap plotting."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    ax = df.traja.plot_heatmap(bins=10)
    assert isinstance(ax, plt.Axes)
    plt.close('all')


def test_plot_speed():
    """Test speed plotting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ax = df.traja.plot_speed()
    assert isinstance(ax, plt.Axes)
    plt.close('all')


def test_plot_trajectory_components():
    """Test comprehensive trajectory component plotting."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = df.traja.plot_trajectory_components(figsize=(10, 8))
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4  # Should have 4 subplots
    plt.close('all')


def test_plot_interactive():
    """Test interactive plotting (requires plotly)."""
    try:
        fig = df.traja.plot_interactive()
        # Check that it returns a plotly figure
        assert hasattr(fig, 'show')
    except ImportError:
        # Skip test if plotly not installed
        import pytest
        pytest.skip("plotly not installed")


def test_plot_interactive_3d():
    """Test interactive 3D plotting."""
    try:
        df_3d = df.copy()
        df_3d['z'] = pd.Series(range(len(df)), dtype=float)
        fig = df_3d.traja.plot_interactive()
        assert hasattr(fig, 'show')
    except ImportError:
        import pytest
        pytest.skip("plotly not installed")
