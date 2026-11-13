"""Astrophysical conversions: magnitude, luminosity, cooling, energy."""

import numpy as np
from numba import njit, prange
from astropy import constants as const
from astropy import units

# astropy ships no stubs for its constants module.
L_SUN_IN_W = const.L_sun.to(units.W).value  # pyright: ignore

# The absolute bolometric magnitude of the Sun, from the IAU 2015 nominal
# values of the solar luminosity and of the zero point (4.74 mag).
M_BOL_SUN = -2.5 * np.log10(
    L_SUN_IN_W / const.L_bol0.to_value(units.W))  # pyright: ignore

# The absolute magnitude of the Sun in each band, and the relative width of
# the band (Delta nu / nu). The magnitudes are from Willmer (2018) Table 3:
# U, B and V are Johnson and K is 2MASS Ks, in the Vega system; g, r, i and
# z are SDSS in the AB system, the same two systems as the TNG
# GFM_StellarPhotometrics. The widths are approximate, after the filter
# compilation of Bessell (1979) and of Schneider et al. (1983), because the
# exact value depends on the filter curve that produced the magnitudes.
_SUN_ABSOLUTE_MAGNITUDE = {
    'U': (5.61, 0.15), 'B': (5.44, 0.22), 'V': (4.81, 0.16),
    'K': (3.27, 0.23), 'g': (5.11, 0.14), 'r': (4.65, 0.14),
    'i': (4.53, 0.16), 'z': (4.50, 0.13),
}


def absoluteMagnitudeToLuminosity(magnitude, band):
    """Turn absolute magnitudes into a band luminosity [W].
    (The band luminosity is nu*L_nu times the relative width of the band, so
    it stays below the bolometric luminosity of the Sun.)
    """
    m_sun, width = _SUN_ABSOLUTE_MAGNITUDE[band]
    # the solar luminosity in this band [W]
    sun = 10.**((m_sun - M_BOL_SUN) / -2.5) * width * L_SUN_IN_W
    return 10.**((np.asanyarray(magnitude) - m_sun) / -2.5) * sun * units.W


# const.G in kpc^3/(Msun s^2): the unit system of the two functions below,
# which read the masses as Msun and the positions as kpc
_G_IN_KPC_MSUN = const.G.to_value(  # pyright: ignore
    units.kpc**3 / (units.M_sun * units.s**2))
_ENERGY_UNIT = units.M_sun * units.kpc**2 / units.s**2


@njit
def _getIthPartGravitationalEnergy(masses, re_coor, i, softening):
    """Return -sum_j m_j/r_ij, the kernel of the energy of the i-th one."""
    total = 0.
    for j in range(len(masses)):
        if j != i:
            dx = re_coor[i, 0] - re_coor[j, 0]
            dy = re_coor[i, 1] - re_coor[j, 1]
            dz = re_coor[i, 2] - re_coor[j, 2]
            total -= masses[j] / ((dx**2. + dy**2. + dz**2.)**0.5 + softening)
    return total


@njit(parallel=True)
def _getPartsGravitationalEnergy(masses, re_coor, softening):
    """Return the kernel of every particle."""
    energy = np.empty(len(masses), dtype=np.float64)
    for i in prange(len(masses)):
        energy[i] = _getIthPartGravitationalEnergy(
            masses, re_coor, i, softening)
    return energy


def getPartsGravitationalEnergy(masses, re_coor, softening=1e-12):
    """Return the gravitational energy of every particle.

    U_i = -G m_i sum_{j != i} m_j / r_ij, in Msun kpc^2/s^2, with the masses
    and the relative positions read as Msun and kpc. Every pair appears once
    from each end, so the sum over i is twice the total energy of the system.
    The softening is added to every separation, so that r never vanishes.
    """
    masses = np.asarray(
        units.Quantity(masses, units.M_sun).to_value(units.M_sun),
        dtype=np.float64)
    re_coor = np.asarray(
        units.Quantity(re_coor, units.kpc).to_value(units.kpc),
        dtype=np.float64)
    kernel = _getPartsGravitationalEnergy(masses, re_coor, softening)
    return _G_IN_KPC_MSUN * masses * kernel * _ENERGY_UNIT


def getGravitationalEnergy(masses, re_coor, softening=1e-12):
    """Return the total gravitational energy of a system of particles.

    U = -G sum_{i < j} m_i m_j / r_ij = (1/2) sum_i U_i, in Msun kpc^2/s^2:
    the 1/2 cancels the double counting of the double sum behind U_i.
    """
    energy = getPartsGravitationalEnergy(masses, re_coor, softening)
    return 0.5 * np.sum(energy)


# the recipes that getCoolingTime() knows
COOLING_TIME_CONVENTIONS = ('zinger20', 'nelson')


def getCoolingTime(
        specific_energy, density, cooling_rate,
        hydrogen_massfrac=0.76, convention='zinger20'):
    """Return the net cooling time of gas cells [Gyr]; NaN where heating.

    'zinger20' is t_cool = U/(n_i n_e Lambda) of Zinger et al. (2020, MNRAS
    499, 768, Eq. 11), whose 1.37 prefactor takes the mean molecular weight
    as 0.58, the value of a fully ionised primordial gas. 'nelson' is the
    n_H^2 recipe of the reference script at
    https://www.tng-project.org/data/forum/topic/550/; its prefactor is
    1 / (1.37 X_H^2), so the ratio of the two is n_i n_e / n_H^2.
    """
    if convention == 'zinger20':
        # mu m_p^2 / ((1 + eps) chi X_H), taken as 1.37 by that section
        factor = 1.37 * const.m_p**2  # pyright: ignore
    elif convention == 'nelson':
        factor = const.m_p**2 / np.asanyarray(  # pyright: ignore
            hydrogen_massfrac)**2
    else:
        raise ValueError(
            f'convention must be one of {COOLING_TIME_CONVENTIONS}, '
            f'not {convention!r}')
    # the cooling rate is negative where the gas cools
    with np.errstate(divide='ignore', invalid='ignore'):
        t_cool = (specific_energy / (density * np.abs(cooling_rate))
                  * factor).to(units.Gyr)
    # a cell with net heating has no cooling time
    return np.where(cooling_rate < 0., t_cool.value, np.nan) * units.Gyr


def internalEnergyToTemperature(internal_energy, electron_abundance,
                                hydrogen_massfrac=0.76):
    """Turn the InternalEnergy of a TNG gas cell into a temperature [K].

    The recipe is the one of https://www.tng-project.org/data/docs/faq/#gen6:
    mu = 4 / (1 + 3 X_H + 4 X_H x_e) m_p, with gamma = 5/3 and X_H = 0.76.
    """
    mass_proton = np.float64(1.672622e-24)  # cgs
    gamma = 5. / 3.
    boltzmann = np.float64(1.380650e-16)  # cgs (erg/K)
    # unit system: the code units of a TNG run, in cgs; the names below are
    # the AREPO parameters that set them (see parameters.txt of a run)
    UnitLength_in_cm = 3.085678e21  # 1.0 kpc
    UnitMass_in_g = 1.989e43  # 1.0e10 solar masses
    UnitVelocity_in_cm_per_s = 1.0e5  # 1 km/sec
    # the unit of time is the unit of length over the unit of velocity, which
    # makes UnitEnergy / UnitMass 1e10, as that page says (its parenthetical
    # "UnitTime is 1 Gyr" would instead give 9.56e9)
    UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
    UnitEnergy_in_cgs = (
        UnitMass_in_g * UnitLength_in_cm**2.0 / UnitTime_in_s**2.0)
    # mean molecular weight, in grams
    meanmolwt = (4.0 / (1.0 + 3.0 * hydrogen_massfrac
                        + 4.0 * hydrogen_massfrac * electron_abundance))
    meanmolwt *= mass_proton
    # temperature (K), in the order of the formula of that page
    temp = (internal_energy * (gamma - 1.0) / boltzmann
            * UnitEnergy_in_cgs / UnitMass_in_g * meanmolwt)
    return temp
