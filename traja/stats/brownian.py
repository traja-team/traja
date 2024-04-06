import numpy as np
from scipy.stats import norm


class Brownian:
    """
    Brownian: Generate brownian motion. Remembers the last position.

    This class caches a large number of samples drawn from a normal
    distribution to compute noise faster.

    Usage: brownian = Brownian(x0=0);
           Brownian()  # Yields x0 + x1 where x1 ~ N(0, 1)
           Brownian()  # Yields x0 + x1 + x2 where x2 ~ N(0, 1)

    Parameters:
    -----------
        x0: Initial x position.
        mean_value: Bias (drift) of the random walk.
        variance: Size of random walk steps.
        length: Number of samples to generate.
        dt: delta-time between every step.
    """

    def __init__(self, x0=0, mean_value=0, variance=1, dt=1.0, length=100000):
        assert (
            type(x0) == float or type(x0) == int or x0 is None
        ), "Expect a float or None for the initial value"

        self._x0 = float(x0)

        # DO NOT modify these values once the class is initialised. The behaviour would
        # be unpredictable
        self._mean_value = mean_value
        self._variance = variance
        self._dt = dt
        self._length = length

        self._index = 0

        self._generate_noise()

    def _generate_noise(self):
        x0 = np.asarray(self._x0)

        # Generate self._length samples of noise
        r = norm.rvs(
            loc=self._mean_value,
            scale=self._variance * np.sqrt(self._dt),
            size=self._length,
        )
        out = np.empty(r.shape)

        # This computes the Brownian motion by forming the cumulative sum of
        # the random samples.
        np.cumsum(r, axis=-1, out=out)

        self._random_walk = out

    def __call__(self):
        assert self._index < self._length, "Random walk is out of samples!"

        sample = self._random_walk[self._index]
        self._index += 1

        return sample

    def __len__(self):
        return len(self._random_walk)
