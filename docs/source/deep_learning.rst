Deep Learning Integration
==========================

Traja provides production-ready features for training neural networks on trajectory data, including data augmentation, sequence processing, feature extraction, and PyTorch integration.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

The deep learning features in traja enable:

* **Data Augmentation** - Create training variations for robust models
* **Sequence Processing** - Standardize trajectory lengths for batching
* **Feature Extraction** - Generate ML-ready features automatically
* **PyTorch Integration** - Seamless tensor conversion
* **Dataset Utilities** - Train/val/test splitting with reproducibility

All features work with both 2D and 3D trajectories.

Data Augmentation
-----------------

Data augmentation is essential for training deep learning models that generalize well. Traja provides five augmentation methods:

Rotation
~~~~~~~~

Rotate trajectories for rotation-invariant models:

.. code-block:: python

   import traja

   df = traja.generate(n=100)

   # Rotate by specific angle
   rotated = df.traja.augment_rotate(angle=45)

   # Random rotation (0-360 degrees)
   rotated = df.traja.augment_rotate()

Gaussian Noise
~~~~~~~~~~~~~~

Add noise for robustness to measurement errors:

.. code-block:: python

   # Add 10% noise relative to coordinate range
   noisy = df.traja.augment_noise(sigma=0.1)

Time Reversal
~~~~~~~~~~~~~

Reverse trajectory temporally when direction doesn't matter:

.. code-block:: python

   reversed_traj = df.traja.augment_reverse()

Scaling
~~~~~~~

Scale coordinates for scale-invariant models:

.. code-block:: python

   # Scale by specific factor
   scaled = df.traja.augment_scale(factor=1.5)

   # Random scaling (0.8-1.2x)
   scaled = df.traja.augment_scale()

Subsampling
~~~~~~~~~~~

Subsample for different temporal resolutions:

.. code-block:: python

   # Keep every 3rd point
   subsampled = df.traja.augment_subsample(step=3)

   # Random step (2-5)
   subsampled = df.traja.augment_subsample()

Sequence Processing
-------------------

For batching variable-length trajectories, you need consistent sequence lengths.

Padding
~~~~~~~

Extend trajectories to target length:

.. code-block:: python

   # Pad with edge mode (repeat last value)
   padded = df.traja.pad_trajectory(target_length=200, mode='edge')

   # Pad with zeros
   padded = df.traja.pad_trajectory(target_length=200, mode='constant')

   # Pad with linear extrapolation
   padded = df.traja.pad_trajectory(target_length=200, mode='linear')

Truncation
~~~~~~~~~~

Shorten trajectories to target length:

.. code-block:: python

   # Keep first N points
   truncated = df.traja.truncate_trajectory(target_length=100, mode='end')

   # Keep last N points
   truncated = df.traja.truncate_trajectory(target_length=100, mode='start')

   # Random starting point
   truncated = df.traja.truncate_trajectory(target_length=100, mode='random')

Normalization
~~~~~~~~~~~~~

Center and scale coordinates for better convergence:

.. code-block:: python

   # Center and scale (mean=0, std=1)
   normalized = df.traja.normalize_trajectory(scale=True, center=True)

   # Center only
   normalized = df.traja.normalize_trajectory(scale=False, center=True)

   # Scale only
   normalized = df.traja.normalize_trajectory(scale=True, center=False)

Feature Extraction
------------------

Automatically extract ML-ready features:

.. code-block:: python

   features = df.traja.extract_features()

   # Returns DataFrame with columns:
   # - displacement: Step-wise displacement
   # - speed: Instantaneous speed (if time available)
   # - turn_angle: Turn angle between steps (2D only)
   # - heading: Direction of movement (2D only)
   # - acceleration: Rate of speed change (if time available)

For 3D trajectories:

.. code-block:: python

   df_3d = traja.TrajaDataFrame({'x': [0,1,2], 'y': [0,1,2], 'z': [0,1,2]})
   features = df_3d.traja.extract_features()

   # Additional columns:
   # - displacement_xy: 2D displacement
   # - displacement_z: Vertical displacement

PyTorch Integration
-------------------

Convert trajectories to PyTorch tensors:

.. code-block:: python

   # Requires: pip install torch

   tensor = df.traja.to_tensor()
   # Returns: torch.Tensor of shape (n_points, 2) for 2D

   # For 3D trajectories
   tensor = df_3d.traja.to_tensor()
   # Returns: torch.Tensor of shape (n_points, 3)

   # Specify columns explicitly
   tensor = df.traja.to_tensor(columns=['x', 'y'])

If PyTorch is not installed, returns numpy array with a warning.

Dataset Splitting
-----------------

Split trajectory lists into train/validation/test sets:

.. code-block:: python

   # Create multiple trajectories
   trajectories = [traja.generate(n=100) for _ in range(50)]

   # Split into train/val/test
   train, val, test = traja.trajectory.train_test_split(
       trajectories,
       train_size=0.7,
       val_size=0.15,
       test_size=0.15,
       shuffle=True,
       random_state=42  # For reproducibility
   )

   print(len(train))  # 35
   print(len(val))    # 7
   print(len(test))   # 8

Complete Pipeline Example
--------------------------

Combining all features for a production pipeline:

.. code-block:: python

   import traja
   import numpy as np

   def preprocess_trajectory(traj, target_length=100, augment=True):
       """Complete preprocessing pipeline for deep learning."""

       # 1. Normalize
       traj = traj.traja.normalize_trajectory()

       # 2. Standardize length
       if len(traj) < target_length:
           traj = traj.traja.pad_trajectory(target_length, mode='edge')
       elif len(traj) > target_length:
           traj = traj.traja.truncate_trajectory(target_length, mode='random')

       # 3. Augmentation (training only)
       if augment:
           if np.random.random() < 0.5:
               traj = traj.traja.augment_rotate()
           if np.random.random() < 0.3:
               traj = traj.traja.augment_noise(sigma=0.05)

       # 4. Convert to tensor
       tensor = traj.traja.to_tensor()

       return tensor

   # Usage
   trajectories = [traja.generate(n=100) for _ in range(100)]
   train, val, test = traja.trajectory.train_test_split(trajectories)

   # Process training data with augmentation
   train_tensors = [preprocess_trajectory(t, augment=True) for t in train]

   # Process validation data without augmentation
   val_tensors = [preprocess_trajectory(t, augment=False) for t in val]

3D Support
----------

All deep learning features support 3D trajectories:

.. code-block:: python

   # Create 3D trajectory
   df_3d = traja.TrajaDataFrame({
       'x': np.random.randn(100),
       'y': np.random.randn(100),
       'z': np.random.randn(100)
   })

   # All operations work the same
   rotated_3d = df_3d.traja.augment_rotate()  # Rotates x,y; z unchanged
   normalized_3d = df_3d.traja.normalize_trajectory()
   tensor_3d = df_3d.traja.to_tensor()  # Shape: (100, 3)

GPS/Lat-Long Support
--------------------

Work with GPS coordinates:

.. code-block:: python

   # Create trajectory from GPS coordinates
   lat = np.array([40.7128, 40.7228, 40.7328])
   lon = np.array([-74.0060, -74.0000, -73.9940])

   traj = traja.from_latlon(lat, lon)

   # Original GPS coordinates preserved
   print(traj[['lat', 'lon', 'x', 'y']])

   # Now use any DL features
   normalized = traj.traja.normalize_trajectory()
   tensor = traj.traja.to_tensor()

Demo Notebook
-------------

See the complete demo with visualizations in:

.. code-block:: bash

   examples/deep_learning_demo.ipynb

The demo uses the public jaguar tracking dataset and shows:

* All augmentation methods with visualizations
* Sequence processing examples
* Feature extraction and plotting
* Complete preprocessing pipeline
* Train/val/test splitting
* GPS coordinate conversion

Use Cases
---------

These features enable:

* **Trajectory Prediction** - Train LSTM/GRU models to predict future positions
* **Trajectory Classification** - Classify movement patterns (foraging, migrating, etc.)
* **Anomaly Detection** - Detect unusual movement patterns
* **Generative Models** - Generate realistic synthetic trajectories with VAE/GAN
* **Transfer Learning** - Pre-train on one species, fine-tune on another

Visualization
-------------

Traja provides several enhanced visualization methods for trajectory analysis.

Interactive Plots
~~~~~~~~~~~~~~~~~

Create interactive plots with zoom, pan, and rotation (requires plotly):

.. code-block:: python

   # Interactive 2D plot
   fig = df.traja.plot_interactive()
   fig.show()

   # Interactive 3D plot
   df_3d = traja.TrajaDataFrame({'x': x, 'y': y, 'z': z})
   fig = df_3d.traja.plot_interactive()
   fig.show()

Heatmap
~~~~~~~

Visualize time spent in each location:

.. code-block:: python

   ax = df.traja.plot_heatmap(bins=50, cmap='hot')
   plt.show()

Speed and Acceleration
~~~~~~~~~~~~~~~~~~~~~~

Plot speed and acceleration profiles:

.. code-block:: python

   # Speed over time
   ax = df.traja.plot_speed()
   plt.show()

   # Acceleration over time (requires time column)
   ax = df.traja.plot_acceleration()
   plt.show()

Comprehensive Analysis
~~~~~~~~~~~~~~~~~~~~~~

4-panel visualization showing path, components, speed, and displacement:

.. code-block:: python

   fig = df.traja.plot_trajectory_components(figsize=(12, 8))
   plt.show()

Performance Optimization
------------------------

For processing large trajectory datasets, use parallel batch processing.

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Process multiple trajectories in parallel (requires joblib):

.. code-block:: python

   import traja

   # Create many trajectories
   trajectories = [traja.generate(n=1000) for _ in range(100)]

   # Process in parallel using all CPUs
   normalized = traja.trajectory.batch_process(
       trajectories,
       lambda t: t.traja.normalize_trajectory(),
       n_jobs=-1  # Use all CPUs
   )

   # With custom function and arguments
   def preprocess(traj, target_length=500):
       traj = traj.traja.normalize_trajectory()
       if len(traj) > target_length:
           traj = traj.traja.truncate_trajectory(target_length)
       return traj

   processed = traja.trajectory.batch_process(
       trajectories,
       preprocess,
       n_jobs=4,  # Use 4 CPUs
       target_length=500
   )

This can provide significant speedup for large datasets (10-100x faster on multi-core machines).

API Reference
-------------

Data Augmentation
~~~~~~~~~~~~~~~~~

.. currentmodule:: traja.accessor

.. autosummary::
   :toctree: reference/

   TrajaAccessor.augment_rotate
   TrajaAccessor.augment_noise
   TrajaAccessor.augment_reverse
   TrajaAccessor.augment_scale
   TrajaAccessor.augment_subsample

Sequence Processing
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   TrajaAccessor.pad_trajectory
   TrajaAccessor.truncate_trajectory
   TrajaAccessor.normalize_trajectory

Feature Extraction
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: reference/

   TrajaAccessor.extract_features
   TrajaAccessor.to_tensor

Dataset Utilities
~~~~~~~~~~~~~~~~~

.. currentmodule:: traja.trajectory

.. autosummary::
   :toctree: reference/

   train_test_split
   batch_process

Visualization Methods
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: traja.accessor

.. autosummary::
   :toctree: reference/

   TrajaAccessor.plot_interactive
   TrajaAccessor.plot_heatmap
   TrajaAccessor.plot_speed
   TrajaAccessor.plot_acceleration
   TrajaAccessor.plot_trajectory_components
