"""Conversions between magnitude and luminosity."""

from astropy import constants as const
from astropy import units

# astropy ships no stubs for its constants module.
L_SUN_IN_W = const.L_sun.to(units.W).value  # pyright: ignore


def absoluteMagnitudeToLuminosity(mag_arr):
    """Convert an absolute V-band magnitude [mag] to a luminosity [W]."""
    M_1 = mag_arr
    M_2 = 4.83  # the absolute magnitude of the Sun
    L_1 = 10.**((M_1 - M_2) / (-2.5)) * L_SUN_IN_W
    return L_1
