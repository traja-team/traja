#! /usr/local/env python3
import math
import numpy as np
import scipy

def stylize_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)


def shift_xtick_labels(xtick_labels, first_index=None):
    for idx, x in enumerate(xtick_labels):
        label = x.get_text()
        xtick_labels[idx].set_text(str(int(label) + 1))
        if first_index is not None:
            xtick_labels[0] = first_index
    return xtick_labels

def fill_in_traj(trj):
    # FIXME: Implement
    return trj

def smooth_sg(trj, w = None, p = 3):
    """Savitzky-Golay filtering."""
    if w is None:
        w = p + 3 - p % 2

    if (w % 2 != 1):
        raise Exception(f"Invalid smoothing parameter w ({w}): n must be odd")
    trj.x = scipy.signal.savgol_filter(trj.x, window_length = w, polyorder=p, axis=0)
    trj.y = scipy.signal.savgol_filter(trj.y, window_length = w, polyorder=p, axis=0)
    trj = fill_in_traj(trj)
    return trj

def angles(trj, lag = 1, compass_direction = None):
    trj['angle'] = np.rad2deg(np.arccos(np.abs(trj['dx']) / trj['distance']))
    # Get heading from angle
    mask = (trj['dx'] > 0) & (trj['dy'] >= 0)
    trj.loc[mask, 'heading'] = trj['angle'][mask]
    mask = (trj['dx'] >= 0) & (trj['dy'] < 0)
    trj.loc[mask, 'heading'] = -trj['angle'][mask]
    mask = (trj['dx'] < 0) & (trj['dy'] <= 0)
    trj.loc[mask, 'heading'] = -(180 - trj['angle'][mask])
    mask = (trj['dx'] <= 0) & (trj['dy'] > 0)
    trj.loc[mask, 'heading'] = (180 - trj['angle'])[mask]
    trj['turn_angle'] = trj['heading'].diff()
    # Correction for 360-degree angle range
    trj.loc[trj.turn_angle >= 180, 'turn_angle'] -= 360
    trj.loc[trj.turn_angle < -180, 'turn_angle'] += 360

def step_lengths(trj):
    """Length of the steps of `trj`."""
    raise NotImplementedError()


def polar_to_z(r, theta):
    """Converts polar coordinates `z` and `theta` to complex number `z`."""
    return r * np.exp(1j * theta)


def cartesian_to_polar(xy):
    """Convert np.array `xy` to polar coordinates `r` and `theta`."""
    assert xy.ndim == 2, f"Dimensions are {xy.ndim}, expecting 2"
    x, y = np.split(xy,[-1], axis=1)
    x, y = np.squeeze(x), np.squeeze(y)
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return r, theta

def expected_sq_displacement(trj, n = None, eqn1= True, compass_direction = None):
    # TODO: Fix and test implementation
    sl = step_lengths(trj)
    ta = angles(trj, compass_direction = compass_direction)
    l = np.mean(sl)
    l2 = np.mean(sl ^ 2)
    c = np.mean(np.cos(ta))
    s = np.mean(np.sin(ta))
    s2 = s ^ 2

    if eqn1:
        # Eqn 1
        alpha = np.arctan2(s, c)
        gamma = ((1 - c)^2 - s2) * np.cos((n + 1) * alpha) - 2 * s * (1 - c) * np.sin((n + 1) * alpha)
        esd = n * l2 + 2 * l^2 * ((c - c^2 - s2) * n  - c) / ((1 - c)^2 + s2) + 2 * l^2 * ((2 * s2 + (c + s2) ^ ((n + 1) / 2)) / ((1 - c)^2 + s2)^2) * gamma
        return abs(esd)
    else:
        # Eqn 2
        esd = n * l2 + 2 * l ^ 2 * c / (1 - c) * (n - (1 - c ^ n) / (1 - c))
        return esd