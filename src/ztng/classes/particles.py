"""Hold the particles of one system and measure them."""

import warnings
from typing import Any

import numpy as np
from astropy import constants as const
from astropy import units

from ..tools import (
    divide,
    getContainCertainFractionRadius,
    getContainCertainValueRadius,
    getRotationTransformationMatrix,
)

G = const.G  # pyright: ignore


class Particles:

    def __init__(self, masses, coordinates, velocities,
                 center_position=None, center_velocity=None,
                 enclosed_mass=None):
        masses = np.asanyarray(masses)
        coordinates = np.asanyarray(coordinates)
        velocities = np.asanyarray(velocities)
        self.count = len(masses)
        lengths = (len(masses), len(coordinates), len(velocities))
        if len(set(lengths)) != 1:
            raise ValueError(
                'masses, coordinates and velocities must have the same '
                f'length, not {lengths[0]}, {lengths[1]} and {lengths[2]}.')
        if self.count == 0:
            raise ValueError('Particles needs at least one particle.')
        self.masses = masses
        self.coordinates = coordinates
        self.velocities = velocities
        if center_position is None:
            self.center_position = np.average(
                coordinates, axis=0, weights=masses)
        else:
            self.center_position = center_position
        if center_velocity is None:
            self.center_velocity = np.average(
                velocities, axis=0, weights=masses)
        else:
            self.center_velocity = center_velocity
        # Here the relative position and velocity are plain differences.
        # If the particle system straddles the edge of the simulation box,
        # wrap the coordinates first, e.g. with processingPbcCoord.
        self.re_coor = self.coordinates - self.center_position
        self.re_vel = self.velocities - self.center_velocity
        self.re_dist = np.linalg.norm(self.re_coor, axis=1)
        # The mass enclosed within the radius of every particle, M(<r).
        # It counts every component of the galaxy inside that radius, not
        # the particles of this instance: an instance built from stars
        # only still needs the gas and the dark matter inside each
        # stellar radius.
        self.enclosed_mass = enclosed_mass

    def getContainCertainMassRadius(self, target_mass):
        """Get the radius that contains a given mass."""
        return getContainCertainValueRadius(
            self.re_dist, self.masses, target_mass)

    def getContainCertainMassFracRadius(self, target_mass_frac):
        """Get the radius that contains a given mass fraction."""
        return getContainCertainFractionRadius(
            self.re_dist, self.masses, target_mass_frac)

    def getHalfMassRadius(self):
        """Get the radius that contains half of the mass."""
        return self.getContainCertainMassFracRadius(0.5)

    def getAngularMomentum(self, specific=True, stack=True):
        """Get the angular momentum, per unit mass if specific."""
        # The specific angular momentum of one particle
        # is the cross product itself.
        per_part_spec_ang_mom = np.cross(self.re_coor, self.re_vel)
        per_part_ang_mom = self.masses[:, np.newaxis] * per_part_spec_ang_mom
        # the total specific angular momentum
        if stack and specific:
            ang_mom = np.sum(per_part_ang_mom, axis=0) / np.sum(self.masses)
        # the total angular momentum
        elif stack:
            ang_mom = np.sum(per_part_ang_mom, axis=0)
        # the specific angular momentum for every particle
        elif specific:
            ang_mom = per_part_spec_ang_mom
        # the angular momentum for every particle
        else:
            ang_mom = per_part_ang_mom
        return ang_mom

    def getPerParticleJz(self):
        """Get jz of every particle.
        Here z is the direction of the total angular momentum.
        """
        sys_ang_mom = self.getAngularMomentum(specific=True, stack=True)
        sys_ang_mom_norm = np.linalg.norm(sys_ang_mom)
        # Test the divisor for an exact zero on purpose: a tolerance would
        # reject a small but well defined angular momentum.
        if sys_ang_mom_norm == 0.:  # noqa: RUF069
            warnings.warn(
                'The total angular momentum is zero, so every jz is NaN.')
        per_part_spec_ang_mom = self.getAngularMomentum(
            specific=True, stack=False)
        per_part_jz = np.matmul(
            per_part_spec_ang_mom, divide(sys_ang_mom, sys_ang_mom_norm))
        return per_part_jz

    def getPerParticleJzSign(self):
        """Get the sign of jz of every particle."""
        per_part_jz = self.getPerParticleJz()
        per_part_jz_sign = np.full(self.count, np.nan, dtype=float)
        per_part_jz_sign[per_part_jz >= 0.] = 1.
        per_part_jz_sign[per_part_jz < 0.] = -1.
        return per_part_jz_sign

    def getCircularities(self):
        """Get jz / j_circ of every particle.
        (Scannapieco et al. 2009, 2012.)
        """
        if self.enclosed_mass is None:
            raise ValueError(
                'Circularity needs the mass enclosed within the radius of '
                'every particle, not the sum of the masses of this '
                'instance.')
        if not (isinstance(self.enclosed_mass, units.Quantity)
                and isinstance(self.re_dist, units.Quantity)):
            raise TypeError(
                'enclosed_mass and the coordinates must be astropy '
                'quantities in physical units, because j_circ = '
                'sqrt(G M(<r) r) is computed from G, which carries units.')
        per_part_jz = self.getPerParticleJz()
        # The circular velocity is sqrt(G * M(<r) / r), and
        # the circular angular momentum is r * v_circ, i.e. sqrt(G * M(<r) * r)
        enclosed = G * self.enclosed_mass * self.re_dist
        per_part_jcirc: Any = np.sqrt(enclosed)
        circularities: Any = divide(per_part_jz, per_part_jcirc)
        return circularities.to(units.dimensionless_unscaled).value

    def getKinematicBulgeFraction(self):
        """Get twice the mass fraction of particles with jz < 0.
        (Scannapieco et al. 2009, 2012.)
        """
        per_part_jz_sign = self.getPerParticleJzSign()
        # No finite sign means the total angular momentum is zero.
        if np.all(~np.isfinite(per_part_jz_sign)):
            return np.nan
        cond = (per_part_jz_sign < 0.)
        bulge_mass = np.sum(self.masses[cond]) * 2.
        total_mass = np.sum(self.masses)
        if total_mass > 0.:
            bulge_mass_frac = bulge_mass / total_mass
            return (bulge_mass_frac * units.dimensionless_unscaled).value
        else:
            return np.nan

    def getKinematicDiscFraction(self, circ_lower_limit=0.7):
        """Get the mass fraction of particles above the circularity limit.
        (Scannapieco et al. 2009, 2012.)
        """
        circularities = self.getCircularities()
        # No finite sign means the total angular momentum is zero.
        if np.all(~np.isfinite(circularities)):
            return np.nan
        cond = (circularities > circ_lower_limit)
        disc_mass = np.sum(self.masses[cond])
        total_mass = np.sum(self.masses)
        if total_mass > 0.:
            disc_mass_frac = disc_mass / total_mass
            return (disc_mass_frac * units.dimensionless_unscaled).value
        else:
            return np.nan

    def getKinematicKappa(self):
        """Get the rotational kinetic energy over the total kinetic energy.
        (Sales et al. 2012.)
        """
        # The rotational kinetic energy is 1/2 * m * (jz / r)^2, and
        # the total kinetic energy is 1/2 * m * v^2.
        # the kappa is the fraction of the rotational kinetic energy,
        # i.e. K_rot / K.
        per_part_jz = self.getPerParticleJz()
        per_part_r = self.re_dist
        per_part_v = np.linalg.norm(self.re_vel, axis=1)
        per_part_m = self.masses
        per_part_vphi = divide(per_part_jz, per_part_r)
        K_rot = np.nansum((1. / 2.) * per_part_m
                          * np.power(per_part_vphi, 2.))
        K = np.nansum((1. / 2.) * per_part_m * np.power(per_part_v, 2.))
        if K > 0.:
            kappa = K_rot / K
            return (kappa * units.dimensionless_unscaled).value
        else:
            return np.nan

    def getRotationTransformationMatrix(self):
        """Get the matrix that rotates these particles to a face-on view.
        np.dot(vector, matrix) gives the vector in the new frame.
        """
        ang_mom_vector = self.getAngularMomentum()
        trans_matrix = getRotationTransformationMatrix(ang_mom_vector)
        return trans_matrix
