"""
Process the data of a certain halo in the IllustrisTNG simulations.
"""


import numpy as np
import warnings
from functools import cached_property
from astropy import units as u
from astropy import constants as const
import illustris_python as il
from ..tools import getPartsGravitationalEnergy
from ..tools import haloVelocityUnit
from ..tools import lengthUnit
from ..tools import massUnit
from ..tools import processingPbcRelativeCoord
from ..accessdata.utils import getSimBasePath
from ..accessdata.online import getSimMeta
from ..accessdata.auto import getSnapHeader
from ..suppldata.local import getHaloStructure
from .group import Group
from .group import _NAMED_RADII
from .groupdataaccess import HaloDataAccess


# Re is the stellar half-mass radius of a subhalo, which a FoF halo has not
_NOT_ALLOWED_NAMED_RADII = tuple(
    name for name, (base, _) in _NAMED_RADII.items() if base == 'Re_ckpch')


class Halo(HaloDataAccess, Group):

    def __init__(self, simName, snapnum, fofid):
        HaloDataAccess.__init__(self, simName, snapnum, fofid)
        Group.__init__(self, simName, snapnum)

    # OOP: Encapsulation, Inheritance, Polymorphism

    # basic

    @cached_property
    def center_position(self):
        """ The center of the halo [ckpc/h]. """
        return np.array(self.info['GroupPos'])

    @cached_property
    def center_velocity(self):
        """ The center velocity of the halo [km/s]. """
        return np.array(self.info['GroupVel']) * haloVelocityUnit(
            self.getScaleFactor()).to_value(u.km / u.s)

    @cached_property
    def m200c(self):
        """ The halo mass within R200c [1e10 Msun/h]. """
        return self.info['Group_M_Crit200']

    @cached_property
    def r200c(self):
        """ The radius where the mean density is 200 times the critical
        density of the Universe [ckpc/h].
        """
        return self.info['Group_R_Crit200']

    @cached_property
    def v200c(self):
        """ The circular velocity at R200c [km/s]. """
        mass = self.m200c * massUnit(self.cosmology.h)
        radius = self.r200c * lengthUnit(
            self.cosmology.h, self.header['Time'])
        velocity = np.sqrt(const.G * mass / radius)  # pyright: ignore
        return velocity.to(u.km / u.s).value

    # particles: load

    def getParts(self, partType, **kwargs):
        """ Get the basic data of particles. """
        within = kwargs.get('within_ckpch')
        if isinstance(within, str) and within in _NOT_ALLOWED_NAMED_RADII:
            raise ValueError(
                f'within_ckpch={within!r}: Re is defined as the stellar '
                'half-mass radius of a subhalo, and a FoF halo has none')
        if 'Re_ckpch' in kwargs:
            raise ValueError(
                'Re_ckpch: Re is defined as the stellar half-mass radius of a '
                'subhalo, and a FoF halo has none')
        kwargs.setdefault('R200_ckpch', self.r200c)
        # OOP: Polymorphism
        data = Group.getParts(self, partType=partType, **kwargs)
        return data

    # particles: mass fraction

    def getStellarMassFracWithinR200(self):
        """ Get the stellar mass fraction within R200 [dimensionless].
        (The stars exclude the wind-phase gas, as in SubhaloMassType, and the
        halo mass in the denominator is Group_M_Crit200.)
        """
        return (self.getPartsMass('stars', onlyRealStars=True,
                                  within_ckpch=self.r200c)
                / self.m200c)

    @cached_property
    def stellar_mass_frac(self):
        """ The stellar mass fraction within R200. """
        return self.getStellarMassFracWithinR200()

    def getBaryonMassFracWithinR200(self):
        """ Get the baryon mass fraction within R200 [dimensionless].
        (The baryons are those of getBaryonMass, so the wind counts as gas and
        the black holes count too, and the denominator is Group_M_Crit200.)
        """
        return self.getBaryonMass(within_ckpch=self.r200c) / self.m200c

    @cached_property
    def baryon_mass_frac(self):
        """ The baryon mass fraction within R200. """
        return self.getBaryonMassFracWithinR200()

    # particles: basic physics

    def getAllPartsGravitationalEnergy(self, return_only_energy=False):
        """
        Get the gravitational energy of all particles.
        (The sum over the array is twice the total energy, see
        tools.getPartsGravitationalEnergy.)
        """
        parts = self.getAllParts(flatten=True)
        particles = self.getPartsInstance(parts, with_physical_units=True)
        particles.gravitational_energy = getPartsGravitationalEnergy(
            particles.masses, particles.re_coor)
        if return_only_energy:
            return particles.gravitational_energy
        else:
            return particles

    # kinematic

    def getSpinParameter(self):
        """
        Get the spin parameter.
        (For details refer to Bullock et al. 2001)
        """
        M_200 = self.m200c * massUnit(self.cosmology.h)
        R_200 = self.r200c * lengthUnit(
            self.cosmology.h, self.header['Time'])
        V_200 = np.sqrt(const.G * M_200 / R_200)  # pyright: ignore
        J_200 = self.getAllPartsAngularMomentum(
            within_ckpch=self.r200c, specific=False, norm=True)
        lambda_ = J_200 / (np.sqrt(2.) * M_200 * V_200 * R_200)
        lambda_ = lambda_.decompose().value
        return lambda_

    def getVirialParameters(self):
        """
        Get the 'virial' parameters of this FoF halo.
        (For details refer to Zinger et al. 2020 doi:10.1093/mnras/staa2607)
        """
        # 'virial' parameters
        Mvir = (self.m200c * massUnit(self.cosmology.h))
        Rvir = (self.r200c * lengthUnit(
            self.cosmology.h, self.header['Time']))
        Vvir = self.v200c * (u.km / u.s)
        mu = 0.59
        Tvir = ((mu * const.m_p * (Vvir**2))  # pyright: ignore
                / (2. * const.k_B)).to(u.K)  # pyright: ignore
        virial_parameters = {}
        virial_parameters['VirialMass'] = Mvir
        virial_parameters['VirialRadius'] = Rvir
        virial_parameters['VirialVelocity'] = Vvir
        virial_parameters['VirialTemperature'] = Tvir
        return virial_parameters

    # in analogous simulation(s)

    def getMatchedInAnalogousSim(self, analogous_simName,
                                 mass_diff_frac=0.3, pos_diff_kpch=100.,
                                 warning=False):
        """
        Get the matched counterpart in an analogous simulation run.
        (See Wang et al. 2019 for details. The two runs must be analogous,
        i.e. of the same series, so that the box and the initial conditions
        are shared.)
        """
        # the series, and the box that comes with it
        if analogous_simName.split('-')[0] != self.simName.split('-')[0]:
            raise ValueError(
                f'{analogous_simName!r} is not of the series of '
                f'{self.simName!r}')
        boxsize = self.getSimMeta()['boxsize']
        analogous_boxsize = getSimMeta(analogous_simName)['boxsize']
        if analogous_boxsize != boxsize:
            raise ValueError(
                f'{analogous_simName!r} and {self.simName!r} have different '
                f'box sizes, {analogous_boxsize} and {boxsize} ckpc/h')
        header = getSnapHeader(self.simName, self.snapnum)
        # load candidates
        analogous_basePath = getSimBasePath(analogous_simName)
        massType = 'Group_M_Crit200'
        posType = 'GroupPos'
        candidates_info = il.groupcat.loadHalos(
            analogous_basePath, self.snapnum, fields=[massType, posType])
        candidates_id = np.arange(candidates_info['count'])
        # deviation, wrapped to the nearest image
        candidates_re_coor = processingPbcRelativeCoord(
            candidates_info[posType] - self.info[posType], boxsize)
        candidates_re_dist = np.linalg.norm(
            candidates_re_coor, axis=1)  # ckpc/h
        candidates_re_dist = candidates_re_dist * header['Time']  # kpc/h
        candidates_mass_diff_frac = (
            np.abs(candidates_info[massType] - self.info[massType])
            / self.info[massType])
        del candidates_info, candidates_re_coor
        # match, and take the candidate with the smallest mass deviation
        cond = ((candidates_re_dist <= pos_diff_kpch)
                & (candidates_mass_diff_frac <= mass_diff_frac))
        if np.any(cond):
            passing = np.flatnonzero(cond)
            best = passing[np.argmin(candidates_mass_diff_frac[passing])]
            return analogous_simName, self.snapnum, candidates_id[best]
        if warning:
            warnings.warn(
                f'[{self.simName} {self.snapnum:02d} {self.fofid}] has no '
                f'counterpart in {analogous_simName}',
                stacklevel=2)
        return analogous_simName, self.snapnum, np.nan

    # supplementary data catalogs

    def getStructuralProperties(self):
        """
        Get the measurements of halo structural properties of the fof halo.

        Supplementary Data Catalogs
        (q) Halo Structure
        https://www.tng-project.org/data/docs/specifications/#sec5q

        Note:
        All are computed using the dark matter particles that both
        (i) belong to the central/primary Subfind subhalo, and;
        (ii) are within R200c of the halo.
        """
        self.structural_properties = getHaloStructure(
            self.simName, self.snapnum, self.fofid)
