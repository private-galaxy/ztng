"""Units of the TNG data.

See https://www.tng-project.org/data/docs/specifications/
Masses and lengths follow one convention in the group data and in the
particle data, but velocities do not: particle Velocities are stored as
sqrt(a) v, GroupVel (for FoF haloes) as v/a, and SubhaloVel already in km/s.
"""

import numpy as np
from astropy import units


def massUnit(hubble_constant):
    """1e10 Msun/h, for group masses and particle masses alike."""
    return 1e10 * units.Msun / hubble_constant


def lengthUnit(hubble_constant, scale_factor):
    """ckpc/h in kpc: comoving, so the scale factor makes it physical."""
    return scale_factor * units.kpc / hubble_constant


def particleVelocityUnit(scale_factor):
    """The stored particle velocity is in sqrt(a) km/s."""
    return np.sqrt(scale_factor) * units.km / units.s


def subhaloVelocityUnit():
    """SubhaloVel is stored in km/s, so there is no factor to apply."""
    return units.km / units.s


def haloVelocityUnit(scale_factor):
    """GroupVel (for FoF haloes) is stored as km/s / a,"""
    return units.km / units.s / scale_factor
