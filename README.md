# traja
Trajectory Analysis in Python

traja is composed of a subclass for pandas DataFrame, along with an accessor for additional operations.

## Installation and setup

Install traja onto your system with `pip install traja`

Import traja into your Python script or via the Python command-line with `import traja`.

## Trajectories with traja

traja stores trajectories in pandas DataFrames, allowing any pandas functions to be used.

Load trajectory:

```python
import traja

traj = traja.from_file('coords.csv')
```

Once a DataFrame is loaded, use the `.traja` accessor to access the visualization and analysis methods:

```python
traj.traja.plot()
```

## Demo

Coming soon.

![dvc_screenshot](dvc_screenshot.png)

Details coming soon.
