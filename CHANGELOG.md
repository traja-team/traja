# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **BREAKING**: Updated minimum Python version from 3.6 to 3.8 to comply with EOL Python versions
- Updated setup.py classifiers to include Python 3.8-3.12
- Modernized pre-commit configuration with latest hooks (black 24.1.1, isort 5.13.2, flake8 7.0.0, mypy 1.8.0)
- Updated GitHub Actions CI to test against Python 3.8, 3.9, 3.10, 3.11, and 3.12

### Added

- **3D Trajectory Support**: Full support for x, y, z spatial coordinates
  - `TrajaAccessor._has_z()` - Check if trajectory has z coordinate
  - `TrajaAccessor.xyz` property - Access x,y,z coordinates as numpy array
  - `TrajaAccessor.center` - Returns (x, y, z) center for 3D trajectories
  - `TrajaAccessor.bounds` - Returns ((xmin, xmax), (ymin, ymax), (zmin, zmax)) for 3D
  - `distance()` - Calculates 3D Euclidean distance if z exists
  - `calc_displacement()` - Computes 3D displacement if z exists
  - `length()` - Automatically supports 3D via calc_displacement()
  - `plot_3d()` - Now uses actual z coordinates if available (falls back to time as z-axis for 2D data)
- **File Format Support**: Added HDF5 and NumPy file support to `read_file()`
  - `.h5`, `.hdf5`, `.hdf` - Read HDF5 files with pandas
  - `.npy` - Read NumPy array files (2-4 columns: x, y, [z], [time])
  - Better error messages for unsupported formats
- **GPS/Lat-Long Support**: Full support for GPS coordinate workflows
  - `latlon_to_xy()` - Convert GPS coordinates to local x, y in meters (Haversine formula)
  - `from_latlon()` - Create TrajaDataFrame directly from lat/lon arrays
  - Preserves original lat/lon columns alongside x/y coordinates
  - Note: Accuracy best for distances <100km; use pyproj for larger areas
- **Export Methods**: Save trajectories in multiple formats
  - `to_csv()` - Export to CSV with all metadata
  - `to_hdf()` - Export to HDF5 format
  - `to_npy()` - Export to NumPy array (auto-selects x,y or x,y,z)
- **Convenience Properties and Methods**:
  - `TrajaAccessor.speed` property - Quick access to instantaneous speed
  - `TrajaAccessor.displacement` property - Quick access to displacement series
  - `TrajaAccessor.summary()` method - Generate comprehensive trajectory statistics
    - Returns dict with: n_points, center, bounds, distance, length, dimensionality, mean_speed, max_speed, mean_acceleration
- **Deep Learning Integration**: Production-ready features for training neural networks on trajectory data
  - **Data Augmentation Methods**:
    - `augment_rotate()` - Rotate trajectory by angle for rotation-invariant models
    - `augment_noise()` - Add Gaussian noise for robustness to measurement errors
    - `augment_reverse()` - Time-reverse trajectory when temporal direction is unimportant
    - `augment_scale()` - Scale coordinates for scale-invariant models
    - `augment_subsample()` - Subsample for different temporal resolutions
  - **Sequence Processing**:
    - `pad_trajectory()` - Pad to target length with 'edge', 'constant', or 'linear' modes
    - `truncate_trajectory()` - Truncate with 'end', 'start', or 'random' modes
    - `normalize_trajectory()` - Center and scale coordinates for better convergence
  - **PyTorch Integration**:
    - `to_tensor()` - Convert to PyTorch tensors (falls back to numpy if torch unavailable)
  - **Feature Extraction**:
    - `extract_features()` - Generate ML-ready features (displacement, speed, turn_angle, heading, acceleration)
  - **Dataset Utilities**:
    - `train_test_split()` - Split trajectory lists into train/val/test sets with shuffle and random_state support
  - Added comprehensive Jupyter notebook demo: `examples/deep_learning_demo.ipynb`
    - Demonstrates all DL features using public jaguar tracking dataset
    - Complete preprocessing pipeline example
    - Visualization of augmentations and feature extraction
- **Visualization Enhancements**: New plotting methods for better trajectory analysis
  - `plot_interactive()` - Interactive 2D/3D plots with plotly (zoom, pan, rotate)
  - `plot_heatmap()` - 2D heatmap showing time spent in each location
  - `plot_speed()` - Speed profile over time
  - `plot_acceleration()` - Acceleration profile over time
  - `plot_trajectory_components()` - Comprehensive 4-panel visualization (path, components, speed, displacement)
- **Performance Optimizations**: Faster processing for large datasets
  - `batch_process()` - Parallel processing of trajectory lists using joblib
  - Added caching mechanism for expensive computations in TrajaAccessor
  - All visualization methods support both 2D and 3D trajectories
- Added comprehensive `pyproject.toml` with modern build system configuration (PEP 517/518 compliant)
- Added mypy configuration for gradual type checking adoption
- Added isort configuration for consistent import ordering
- Added pytest and coverage configuration in pyproject.toml
- Added type hints to core functions in `frame.py`, `trajectory.py`, and `core.py`
- Added proper docstring to `is_datetime_or_timedelta_dtype()` function
- Enhanced pre-commit hooks with additional checks (trailing-whitespace, check-yaml, check-json, etc.)

### Documentation

- Updated `README.rst` to reflect Python 3.8+ requirement
- Added Deep Learning Integration section to `README.rst` with examples
- Created comprehensive `docs/source/deep_learning.rst` documentation page:
  - Complete API reference for all DL features
  - Code examples for every feature
  - Complete preprocessing pipeline example
  - Use cases and best practices
- Added deep learning page to documentation index
- Updated `docs/source/install.rst` with correct Python version requirement
- Updated `docs/source/contributing.rst` with:
  - Python 3.8+ requirement
  - Type hints guidelines for contributors
  - Pre-commit hooks usage instructions
  - Description of automated code quality checks (black, isort, flake8, mypy)
- Created `examples/README.md` documenting the deep learning demo notebook

### Fixed

- Fixed deprecated `pd.datetime` usage, replaced with `pd.to_datetime()` (fixes pandas 2.0+ compatibility)
- Fixed lambda closure bug in `parsers.py:126` by using default argument pattern
- **Performance**: Replaced inefficient `np.append()` calls with `np.concatenate()` or list comprehensions:
  - `trajectory.py:488` - Used list comprehension for angle accumulation (major speedup in `calc_flow_angles()`)
  - `trajectory.py:679` - Replaced `np.append()` with `np.concatenate()` in trajectory generation
  - `trajectory.py:1199,1205` - Used `np.concatenate()` for frame boundary handling
  - `plotting.py:1077` - Replaced `np.append()` with `np.concatenate()` in run-length encoding
- **Critical Bug**: Fixed missing `inplace=True` in DataFrame rename operations (silent failures):
  - `parsers.py:153,161` - Fixed column renaming in `read_file()`
  - `trajectory.py:274,278` - Fixed column renaming in `from_xy()`
  - `trajectory.py:1256,1259,1260,1266` - Fixed Series rename and DataFrame replace not being assigned
- Replaced 18 generic `Exception` raises with specific exception types:
  - `ValueError` for invalid parameter values (7 instances)
  - `KeyError` for missing required columns (2 instances)
  - `TypeError` for type mismatches (3 instances)
  - `FileNotFoundError` for missing files (2 instances)
- Improved error messages to be more descriptive and actionable
- Enhanced `is_datetime_or_timedelta_dtype()` to accept both Series and Index types (addresses issue #103)

### Technical Improvements

- Better type safety with gradual type hint adoption
- Improved code quality with comprehensive linting in pre-commit
- More robust CI/CD pipeline testing across all supported Python versions
- Modern package metadata and build configuration
- Better development workflow with pre-commit hooks

### Migration Notes

- Projects using Python 3.6 or 3.7 must upgrade to Python 3.8+
- Code catching generic `Exception` may need updates to catch specific exception types
- The deprecated pandas datetime API is now fixed and compatible with pandas 2.0+
