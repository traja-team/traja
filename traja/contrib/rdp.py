"""
rdp
~~~
Python implementation of the Ramer-Douglas-Peucker algorithm.
:copyright: 2014-2016 Fabian Hirschmann <fabian@hirschmann.email>
:license: MIT.

Copyright (c) 2014 Fabian Hirschmann <fabian@hirschmann.email>.
With minor modifictions by Justin Shenk (c) 2019.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
"""
from functools import partial
from typing import Union, Callable

import numpy as np
import sys


def pldist(point: np.ndarray, start: np.ndarray, end: np.ndarray):
    """
    Calculates the distance from ``point`` to the line given
    by the points ``start`` and ``end``.
    :param point: a point
    :type point: numpy array
    :param start: a point of the line
    :type start: numpy array
    :param end: another point of the line
    :type end: numpy array
    """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
        np.abs(np.linalg.norm(np.cross(end - start, start - point))),
        np.linalg.norm(end - start),
    )


def rdp_rec(M, epsilon, dist=pldist):
    """
    Simplifies a given array of points.
    Recursive version.
    :param M: an array
    :type M: numpy array
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
    """
    dmax = 0.0
    index = -1

    for i in range(1, M.shape[0]):
        d = dist(M[i], M[0], M[-1])

        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        r1 = rdp_rec(M[: index + 1], epsilon, dist)
        r2 = rdp_rec(M[index:], epsilon, dist)

        return np.vstack((r1[:-1], r2))
    else:
        return np.vstack((M[0], M[-1]))


def _rdp_iter(M, start_index, last_index, epsilon, dist=pldist):
    stk = []
    stk.append([start_index, last_index])
    global_start_index = start_index
    indices = np.ones(last_index - start_index + 1, dtype=bool)

    while stk:
        start_index, last_index = stk.pop()

        dmax = 0.0
        index = start_index

        for i in range(index + 1, last_index):
            if indices[i - global_start_index]:
                d = dist(M[i], M[start_index], M[last_index])
                if d > dmax:
                    index = i
                    dmax = d

        if dmax > epsilon:
            stk.append([start_index, index])
            stk.append([index, last_index])
        else:
            for i in range(start_index + 1, last_index):
                indices[i - global_start_index] = False

    return indices


def rdp_iter(
    M: Union[list, np.ndarray],
    epsilon: float,
    dist: Callable = pldist,
    return_mask: bool = False,
):
    """
    Simplifies a given array of points.
    Iterative version.
    :param M: an array
    :type M: numpy array
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
    :param return_mask: return the mask of points to keep instead
    :type return_mask: bool

    .. note::
        Yanked from Fabian Hirschmann's PyPI package ``rdp``.

    """
    mask = _rdp_iter(M, 0, len(M) - 1, epsilon, dist)

    if return_mask:
        return mask

    return M[mask]


def rdp(
    M: Union[list, np.ndarray],
    epsilon: float = 0,
    dist: Callable = pldist,
    algo: str = "iter",
    return_mask: bool = False,
):
    """
    Simplifies a given array of points using the Ramer-Douglas-Peucker
    algorithm.
    Example:
    >>> from traja.contrib import rdp
    >>> rdp([[1, 1], [2, 2], [3, 3], [4, 4]])
    [[1, 1], [4, 4]]
    This is a convenience wrapper around both :func:`rdp.rdp_iter`
    and :func:`rdp.rdp_rec` that detects if the input is a numpy array
    in order to adapt the output accordingly. This means that
    when it is called using a Python list as argument, a Python
    list is returned, and in case of an invocation using a numpy
    array, a NumPy array is returned.
    The parameter ``return_mask=True`` can be used in conjunction
    with ``algo="iter"`` to return only the mask of points to keep. Example:
    >>> from traja.contrib import rdp
    >>> import numpy as np
    >>> arr = np.array([1, 1, 2, 2, 3, 3, 4, 4]).reshape(4, 2)
    >>> arr
    array([[1, 1],
           [2, 2],
           [3, 3],
           [4, 4]])
    >>> mask = rdp(arr, algo="iter", return_mask=True)
    >>> mask
    array([ True, False, False,  True], dtype=bool)
    >>> arr[mask]
    array([[1, 1],
           [4, 4]])
    :param M: a series of points
    :type M: numpy array with shape ``(n,d)`` where ``n`` is the number of points and ``d`` their dimension
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
    :param algo: either ``iter`` for an iterative algorithm or ``rec`` for a recursive algorithm
    :type algo: string
    :param return_mask: return mask instead of simplified array
    :type return_mask: bool

    .. note::
        Yanked from Fabian Hirschmann's PyPI package ``rdp``.

    """

    if algo == "iter":
        algo = partial(rdp_iter, return_mask=return_mask)
    elif algo == "rec":
        if return_mask:
            raise NotImplementedError('return_mask=True not supported with algo="rec"')
        algo = rdp_rec

    if "numpy" in str(type(M)):
        return algo(M, epsilon, dist)

    return algo(np.array(M), epsilon, dist).tolist()
