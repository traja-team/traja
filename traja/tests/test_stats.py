from traja.stats.brownian import Brownian
import numpy as np

def test_brownian_walk_generates_correct_number_of_samples():
    length = 1000000
    brownian = Brownian(length=length)
    assert(len(brownian) == length)


def test_brownian_motion_with_drift_approximately_sums_to_the_drift():
    length = 1000000
    mean_drift = 0.1
    drift = 0

    brownian = Brownian(length=length, mean_value=mean_drift)

    for i in range(length):
        drift = brownian()

    drift /= length

    np.testing.assert_approx_equal(drift, mean_drift, significant=1)


def test_brownians_with_different_variances_drift_approximately_equally():
    length = 1000000
    mean_drift = -0.9
    variance1 = 0.8
    variance2 = 3.5

    drift1 = 0
    drift2 = 0

    brownian1 = Brownian(length=length, mean_value=mean_drift, variance=variance1)
    brownian2 = Brownian(length=length, mean_value=mean_drift, variance=variance2)

    for i in range(length):
        drift1 = brownian1()
        drift2 = brownian2()

    drift1 /= length
    drift2 /= length

    np.testing.assert_approx_equal(drift1, drift2, significant=1)


def test_brownians_with_different_time_steps_walk_approximately_equally():
    mean_drift = 0.23
    variance = 0.52

    time_step_ratio = 7
    dt1 = 0.1
    dt2 = dt1 * time_step_ratio

    length2 = 200000
    length1 = length2 * time_step_ratio

    drift1 = 0
    drift2 = 0

    brownian1 = Brownian(length=length1, mean_value=mean_drift, variance=variance, dt=dt1)
    brownian2 = Brownian(length=length2, mean_value=mean_drift, variance=variance, dt=dt2)

    for i in range(length1):
        drift1 = brownian1()

    for i in range(length2):
        drift2 = brownian2()

    drift1 /= length1
    drift2 /= length2

    np.testing.assert_approx_equal(drift1, drift2, significant=1)
