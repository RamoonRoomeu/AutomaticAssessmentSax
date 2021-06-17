"""
This file contains bunch of functions that compute different distance metrics, which finally are used to compute
similarity between musical performances. Most of these functions are simple sample to sample distance metrics.
"""
from scipy.spatial import distance


def euclidean_distance(x, y):
    """Computes Euclidean distance between two arrays.

    Parameters
    ----------
    x : (N,) array_like
        Input array
    y : (N,) array_like
        Input array

    Returns
    -------
    distance: double
        Euclidean distance between two arrays `x` and `y`

    """
    return distance.euclidean(x, y)


def manhattan_distance(x, y):
    """Computes Manhattan distance between two arrays.

    Parameters
    ----------
    x : (N,) array_like
        Input array
    y : (N,) array_like
        Input array

    Returns
    -------
    distance: double
        Manhattan distance between two arrays `x` and `y`

    """
    return distance.cityblock(x, y)


def minkowski_distance(x, y, p=1):
    """Computes Minkowski distance between two arrays.

    Parameters
    ----------
    x : (N,) array_like
        Input array
    y : (N,) array_like
        Input array
    p : int
        The order of the norm of the difference :math:`{||x-y||}_p`.

    Returns
    -------
    distance: double
        Minkowski distance between two arrays `x` and `y`

    """
    return distance.minkowski(x, y, p)


def cosine_distance(x, y):
    """Computes Cosine distance between two arrays.

    Parameters
    ----------
    x : (N,) array_like
        Input array
    y : (N,) array_like
        Input array

    Returns
    -------
    distance: double
        Cosine distance between two arrays `x` and `y`
    """
    return distance.cosine(x, y)
