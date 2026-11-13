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
    elif not np.isfinite(target_value):
        # A NaN or an infinite target can never be reached.
        target_radius = np.nan
    elif target_value < 0.:
        # No radius holds a negative amount.
        target_radius = np.nan
    elif target_value > np.sum(parts_value):
        target_radius = np.nan
    elif target_value == 0.:  # noqa: RUF069
        # The sphere of radius zero holds nothing.
        target_radius = 0.
    else:
        order = np.argsort(parts_re_dist)
        cumulative = np.cumsum(parts_value[order])
        index = np.searchsorted(cumulative, target_value, side='left')
        # np.sum() and np.cumsum() differ in the last bits, so a target that
        # equals the total can land one past the last particle
        index = min(index, len(cumulative) - 1)
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


def getEnclosedMass(parts_re_dist, all_re_dist, all_masses):
    """Return M(<r) for each r of parts_re_dist.

    all_re_dist and all_masses describe every particle, so M counts all of
    them, not only the particles of parts_re_dist. The mass carries the unit
    of all_masses.
    """
    order = np.argsort(all_re_dist)
    cumulative = np.cumsum(all_masses[order])
    # side='left' keeps the strict '<', so a particle does not count itself
    index = np.searchsorted(all_re_dist[order], parts_re_dist, 'left')
    return np.where(index > 0, cumulative[np.maximum(index - 1, 0)], 0.)
