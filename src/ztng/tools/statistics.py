"""Statistics of a set of particles."""

import warnings

import numpy as np
from astropy.units import Quantity


def getContainCertainValueRadius(
        parts_re_dist, parts_value, target_value):
    """Return the radius that holds at least target_value, or NaN."""
    if isinstance(parts_re_dist, Quantity):
        radius_unit = parts_re_dist.unit
        parts_re_dist = parts_re_dist.value
    else:
        radius_unit = None
    if len(parts_value) == 0:
        # No particles, so there is no radius.
        target_radius = np.nan
    elif not np.all(np.isfinite(parts_value)):
        warnings.warn('parts_value holds a value that is not finite, so the '
                      'radius is undefined.')
        target_radius = np.nan
    elif target_value > np.sum(parts_value):
        target_radius = np.nan
    else:
        order = np.argsort(parts_re_dist)
        cumulative = np.cumsum(parts_value[order])
        index = np.searchsorted(cumulative, target_value, side='left')
        target_radius = parts_re_dist[order][index]
    if radius_unit is None:
        return target_radius
    return target_radius * radius_unit


def getContainCertainFractionRadius(
        parts_re_dist, parts_value, target_fraction):
    """Return the radius that holds the target fraction of the total."""
    target_value = np.sum(parts_value) * target_fraction
    target_radius = getContainCertainValueRadius(
        parts_re_dist, parts_value, target_value)
    return target_radius
