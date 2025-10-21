# Traja Examples

This directory contains example notebooks demonstrating traja's capabilities.

## Deep Learning Demo

**File**: `deep_learning_demo.ipynb`

A comprehensive demonstration of traja's deep learning features using the public jaguar tracking dataset:

### Features Demonstrated

1. **Data Augmentation**
   - Rotation (rotation-invariant models)
   - Gaussian noise (robustness to measurement errors)
   - Time reversal (temporal invariance)
   - Scaling (scale-invariant models)
   - Subsampling (temporal resolution variation)

2. **Sequence Processing**
   - Padding trajectories to target length
   - Truncating trajectories
   - Normalization (centering and scaling)

3. **Feature Extraction**
   - Automatic ML feature generation
   - Displacement, speed, turn angle, heading, acceleration

4. **PyTorch Integration**
   - Easy conversion to PyTorch tensors
   - Fallback to numpy arrays if PyTorch unavailable

5. **Dataset Utilities**
   - Train/validation/test splitting
   - Reproducible random splits

6. **GPS Support** (Bonus)
   - Convert latitude/longitude to local x,y coordinates
   - Haversine formula for accurate short-range conversion

### Running the Demo

```bash
# Install traja (development mode)
cd /path/to/traja
pip install -e .

# Optional: Install PyTorch for tensor conversion
pip install torch

# Launch Jupyter
jupyter notebook examples/deep_learning_demo.ipynb
```

### Data Source

The demo uses the public jaguar tracking dataset from the traja research repository. The data is automatically downloaded when running the notebook.

### Use Cases

This demo is useful for:
- Training LSTM/GRU models on trajectory data
- Preparing data for trajectory prediction models
- Building trajectory classification systems
- Analyzing animal movement patterns
- Processing GPS tracking data

### Complete Pipeline Example

The notebook includes a complete preprocessing pipeline that combines:
1. Normalization
2. Sequence length standardization (pad/truncate)
3. Random augmentation
4. Tensor conversion

This pipeline is production-ready for deep learning workflows.
