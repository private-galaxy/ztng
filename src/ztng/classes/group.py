"""Process the data of a group object (halo or subhalo)."""


import numpy as np
from functools import cached_property
import hashlib
import warnings
from typing import Any
from astropy import units
from astropy import constants as const
import illustris_python as il
from ..tools import COOLING_TIME_CONVENTIONS
from ..tools import absoluteMagnitudeToLuminosity
from ..tools import getContainCertainFractionRadius
from ..tools import getCoolingTime
from ..tools import getEnclosedMass
from ..tools import internalEnergyToTemperature
from ..tools import lengthUnit
from ..tools import massUnit
from ..tools import particleVelocityUnit
from ..tools import processingPbcRelativeCoord
from ..tools import processingPbcRelativeDist
from ..tools import updateKwargs
from ..accessdata.utils import MAX_SNAPNUM
from ..accessdata.utils import MIN_SNAPNUM
from ..accessdata import online
from ..accessdata import auto
from ..accessdata.auto import getSnapRedshift
from ..accessdata.auto import getSnapLookbackTime
from ..accessdata.utils import inFullSnap
from ..accessdata.utils import getSimBasePath
from ..accessdata.utils import getSavedDataPath
from .particles import Particles


# the fields that every particle type has
_DEFAULT_FIELDS = ['ParticleIDs', 'Coordinates', 'Velocities']

# what getParts adds to those, by the numeric particle type
_BASIC_FIELDS = {
    0: _DEFAULT_FIELDS + ['Masses', 'StarFormationRate'],  # gas
    1: _DEFAULT_FIELDS,  # dm
    4: _DEFAULT_FIELDS + ['Masses', 'GFM_StellarFormationTime'],  # stars/wind
    5: _DEFAULT_FIELDS + ['Masses'],  # bhs
}

# the particle types of a group, in the order that flatten=True lays out
_ALL_PART_TYPES = ('dm', 'gas', 'stars', 'bh')

# the allowed values of the center keyword of getAngularMomentum() and of
# getPartsInstance(): the center of the object, or the center of the particles
_ALLOWED_CENTERS = ('object', 'parts')

# the keywords of getParts() and their defaults
_DEFAULT_GET_PARTS_KWARGS = {
    'verbose': False,  # print the keywords and the cache state
    'fields': [],  # extra fields on top of _BASIC_FIELDS
    'within_ckpch': np.inf,  # radius, range or a name in _NAMED_RADII
    'R200_ckpch': np.nan,  # R200 of the FoF halo, in ckpc/h
    'Re_ckpch': np.nan,  # stellar half-mass radius of the subhalo, in ckpc/h
    'onlyRealStars': False,  # real stars only, for partType='stars'
    'onlySFingGas': False,  # star-forming gas only, for partType='gas'
}

# named radius: a factor of a radius
_NAMED_RADII = {
    'R200': ('R200_ckpch', 1.),
    '0d1R200': ('R200_ckpch', 0.1),
    'Re': ('Re_ckpch', 1.),
    '2Re': ('Re_ckpch', 2.),
    '10Re': ('Re_ckpch', 10.),
}


def _processingWithinCkpch(within_ckpch, R200_ckpch, Re_ckpch):
    """Turn a radius selection into a [lower, upper] range in ckpc/h."""
    if isinstance(within_ckpch, (int, float, np.floating)):
        within_ckpch = [0., float(within_ckpch)]

    elif (isinstance(within_ckpch, list) and len(within_ckpch) == 2):
        within_ckpch = [float(within_ckpch[0]), float(within_ckpch[1])]

    elif (isinstance(within_ckpch, str) and within_ckpch in _NAMED_RADII):
        base_name, factor = _NAMED_RADII[within_ckpch]
        base = {'R200_ckpch': R200_ckpch, 'Re_ckpch': Re_ckpch}[base_name]
        if not np.isfinite(base):
            raise ValueError(
                f'within_ckpch={within_ckpch!r} needs {base_name}, '
                'which this object did not provide')
        if not base * factor > 0.:
            raise ValueError(
                f'within_ckpch={within_ckpch!r} needs {base_name} > 0, '
                f'not {base!r}')
        within_ckpch = [0., base * factor]

    else:
        raise TypeError(
            "'within_ckpch' must be float or 2-items list "
            "or 'R200', '0d1R200', 'Re', '2Re', '10Re'.")

    return within_ckpch


def _selectParts(data, cond):
    """Keep the particles where cond is true, count included."""
    for key in data:
        if key == 'count':
            data[key] = np.sum(cond)
        else:
            data[key] = data[key][cond]


def _flattenAllParts(all_parts):
    """Merge the dicts of all particle types into one, adding 'partType'."""
    data = {'count': 0, 'partType': np.array([], dtype='<U5')}
    for partType in _ALL_PART_TYPES:
        parts = all_parts[partType]
        if parts['count'] == 0:
            continue
        data['count'] += parts['count']
        data['partType'] = np.append(
            data['partType'], np.repeat(partType, parts['count']), axis=0)
        for key in ['ParticleIDs', 'Masses', 'Coordinates', 'Velocities']:
            if key in data:
                data[key] = np.append(data[key], parts[key], axis=0)
            else:
                # a copy, so that the callers cannot rewrite the getParts()
                # cache through the flattened arrays
                data[key] = parts[key].copy()
    # keep the same keys when every particle type is empty
    if data['count'] == 0:
        data['ParticleIDs'] = np.array([], dtype='uint64')
        data['Masses'] = np.array([], dtype=float)
        data['Coordinates'] = np.empty((0, 3))
        data['Velocities'] = np.empty((0, 3))
    # the returned dict will only have the keys of these fields
    return data


def _partsCacheKey(partType, kwargs):
    """Return a unique key for one set of getParts() arguments."""
    key_string = f'getParts()_partType_{partType}'
    key_string += f'_{tuple(sorted(kwargs.items()))}'
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()


class Group:

    # these are provided by the mixins of the Halo and Subhalo classes.
    getSnap: Any
    center_position: Any  # in ckpc/h
    center_velocity: Any  # converted to km/s

    def __init__(self, simName, snapnum):
        self.simName = simName
        self.snapnum = snapnum
        self.basePath = getSimBasePath(simName)
        self.dataPath = getSavedDataPath()
        # NOTE: The cache has no size limit and holds a full set of particle
        # arrays per argument set. Use an LRU cache if memory runs out.
        self._cache = {}

    def getSimMeta(self):
        """ Get the simulation meta-data. """
        return online.getSimMeta(self.simName)

    def getCosmology(self):
        """ Get the cosmological calculation. """
        return online.getSimCosmology(self.simName)

    @cached_property
    def cosmology(self):
        return self.getCosmology()

    def getSnapMeta(self):
        """ Get the snapshot meta-data. """
        return online.getSnapMeta(self.simName, self.snapnum)

    def getHeader(self):
        """ Get the group catalog header. """
        return auto.getSnapHeader(self.simName, self.snapnum)

    @cached_property
    def header(self):
        return self.getHeader()

    # cosmic time

    def getRedshift(self):
        """ Get the redshift. """
        return getSnapRedshift(self.simName, self.snapnum)

    @cached_property
    def redshift(self):
        return self.getRedshift()

    def getLookbackTime(self):
        """ Get the lookback time. """
        return getSnapLookbackTime(self.simName, self.snapnum)

    @cached_property
    def lookbacktime(self):
        return self.getLookbackTime()

    def getScaleFactor(self):
        """ Get the comoving scale factor (a=1/(1+z)). """
        return self.header['Time']

    def inFullSnap(self):
        """ Is this object in the 'full' snapshot?
        (Note that, unlike in Illustris,
        TNG contains two different types of snapshots: 'full' and 'mini'.
        For details refer to
        https://www.tng-project.org/data/docs/specifications/)
        """
        return inFullSnap(self.snapnum)

    # particles: load

    def getParts(self, partType, **kwargs):
        """ Get the basic data of particles.
        (The keywords are the keys of _DEFAULT_GET_PARTS_KWARGS, and any other
        keyword raises a TypeError. The dict is a shallow copy of the cached
        one: keys are safe to add or replace, the arrays are shared. PartType4
        holds the real stars and the wind, told apart by the sign of
        GFM_StellarFormationTime; match the catalog stellar mass with
        onlyRealStars=True, because the catalog counts the wind as gas.)
        """
        # OOP: Inheritance, Polymorphism
        # >> subhalo.getAllParts() will call Subhalo.getParts()
        # >> halo.getAllParts() will call Halo.getParts()

        # default kwargs
        kwargs = updateKwargs(_DEFAULT_GET_PARTS_KWARGS, kwargs)

        # pre-process the kwargs: 'fields' and 'within_ckpch'
        ptNum = il.util.partTypeNum(partType)
        # a flag that does not apply to this type are ignored, with a warning.
        if kwargs['onlyRealStars'] and ptNum != il.util.partTypeNum('stars'):
            warnings.warn(
                f'onlyRealStars has no effect on partType={partType!r}: '
                'only PartType4 holds wind particles.', stacklevel=3)
        if kwargs['onlySFingGas'] and ptNum != il.util.partTypeNum('gas'):
            warnings.warn(
                f'onlySFingGas has no effect on partType={partType!r}: '
                'only the gas cells have a StarFormationRate.', stacklevel=3)
        kwargs['fields'] = list(
            set(kwargs['fields'] + _BASIC_FIELDS.get(ptNum, _DEFAULT_FIELDS)))
        kwargs['within_ckpch'] = _processingWithinCkpch(
            kwargs['within_ckpch'], kwargs['R200_ckpch'], kwargs['Re_ckpch'])

        # verbose
        if kwargs['verbose']:
            print(kwargs)
            # external functions to be called and used
            help(self.getSnap)

        # check whether this set of parameters has been used
        cache_key = _partsCacheKey(
            partType, {key: kwargs[key] for key in kwargs
                       if key != 'verbose'})

        # check if already have cached data for this set of parameters
        if cache_key in self._cache:
            if kwargs['verbose']:
                print('Return the cached result.')
            # a copy, so that the callers cannot rewrite the cached dict
            return dict(self._cache[cache_key])

        # If not, get it (and cache it).

        # raw particles data
        data = self.getSnap(partType, kwargs['fields'])

        # post-processing
        if data['count'] > 0:
            # only real stars? (not wind)
            if ((ptNum == il.util.partTypeNum('stars'))
                    and kwargs['onlyRealStars']):
                _selectParts(data, data['GFM_StellarFormationTime'] > 0.)
            # only star-forming gas?
            elif ((ptNum == il.util.partTypeNum('gas'))
                    and kwargs['onlySFingGas']):
                _selectParts(data, data['StarFormationRate'] > 0.)
            # mass of dark matter paticle
            elif ptNum == il.util.partTypeNum('dm'):
                data['Masses'] = np.full(
                    data['count'], self.getSimMeta()['mass_dm'],
                    dtype=np.float32)

        # only particles within a certain radius range ([within_ckpch] ckpc/h)
        if data['count'] > 0:
            rad_lower_limit = np.min(kwargs['within_ckpch'])  # ckpc/h
            rad_upper_limit = np.max(kwargs['within_ckpch'])  # ckpc/h
            # wrap the offsets into [-box/2, box/2), so that an object across
            # a periodic boundary keeps its members
            box_length = self.getSimMeta()['boxsize']  # ckpc/h
            re_dist = processingPbcRelativeDist(
                data['Coordinates'], self.center_position, box_length)
            cond = ((re_dist >= rad_lower_limit)
                    & (re_dist <= rad_upper_limit))
            _selectParts(data, cond)
            del cond

        # cache the computed result for future use
        self._cache[cache_key] = data
        if kwargs['verbose']:
            print('Return the result and cache it.')

        # a copy, so that the callers cannot rewrite the cached dict
        return dict(data)

    def getAllParts(
            self, within_ckpch=np.inf, flatten=False):
        """ Get basic data for all types of particles.
        (With flatten=True, one dict holds them all.)
        """
        # OOP: Inheritance, Polymorphism
        # >> gourp.getAllParts() will call Group.getParts()
        # >> subhalo.getAllParts() will call Subhalo.getParts()
        all_parts = {}
        for partType in _ALL_PART_TYPES:
            all_parts[partType] = self.getParts(
                partType, within_ckpch=within_ckpch)
        # collapsed into one dict
        if flatten:
            return _flattenAllParts(all_parts)
        return all_parts

    def getPartsInstance(
            self, parts,
            with_physical_units=False,
            input_enclosed_mass=False,
            center='object') -> Any:
        """ Create an instance of the class Particles.

        Note: without with_physical_units, the arrays stay in the units of
        the data: masses in 1e10 Msun/h, positions in ckpc/h, and velocities
        converted to km/s. With input_enclosed_mass, the enclosed mass counts
        all types of particles, not only the ones in 'parts'. With
        center='parts', the center is the mass weighted center of the
        particles and their mean velocity, not the center of the object.
        """
        if 'count' not in parts:
            warnings.warn(
                'parts has no count: taking it from the length of Masses. ',
                stacklevel=2)
            parts['count'] = len(parts['Masses'])
        if int(parts['count']) == 0:
            return parts

        # preparing
        cosmo = self.cosmology
        header = self.header
        center_position = self.center_position  # ckpc/h
        center_velocity = self.center_velocity  # km/s
        masses = parts['Masses']  # 1e10 Msun/h
        coordinates = parts['Coordinates']  # ckpc/h
        velocities = parts['Velocities'] * particleVelocityUnit(header['Time'])
        velocities = velocities.to_value(units.km / units.s)  # km/s
        if 'ParticleIDs' in parts:
            ids = parts['ParticleIDs']
        else:
            ids = None
        enclosed_mass = None

        # the box length wraps the relative positions around the box
        box_length = self.getSimMeta()['boxsize']  # ckpc/h

        # the center of the particles, in the units of this point (ckpc/h)
        if center == 'parts':
            offsets = processingPbcRelativeCoord(
                coordinates - center_position, box_length)
            center_position = center_position + np.average(  # pyright: ignore
                offsets, axis=0, weights=masses)
            center_velocity = center_velocity + np.average(  # pyright: ignore
                velocities - center_velocity, axis=0, weights=masses)
        elif center != 'object':
            raise ValueError(
                f'center must be one of {_ALLOWED_CENTERS}, not {center!r}')

        # calculate the M(<r) for each r
        if input_enclosed_mass:
            all_parts = self.getAllParts(
                within_ckpch=np.inf, flatten=True)
            all_re_dist = processingPbcRelativeDist(
                all_parts['Coordinates'], center_position, box_length)
            parts_re_dist = processingPbcRelativeDist(
                coordinates, center_position, box_length)
            enclosed_mass = getEnclosedMass(
                parts_re_dist, all_re_dist, all_parts['Masses'])  # 1e10 Msun/h

        # convert the units to physical ones, if requested
        if with_physical_units:
            masses = masses * massUnit(cosmo.h)
            coordinates = coordinates \
                * lengthUnit(cosmo.h, header['Time'])
            velocities = velocities * (units.km / units.s)
            center_position = center_position \
                * lengthUnit(cosmo.h, header['Time'])
            center_velocity = center_velocity * (units.km / units.s)
            box_length = box_length \
                * lengthUnit(cosmo.h, header['Time'])
            if input_enclosed_mass:
                enclosed_mass = enclosed_mass * massUnit(cosmo.h)

        # create the Particles instance
        parts = Particles(masses, coordinates, velocities,
                          center_position, center_velocity,
                          enclosed_mass=enclosed_mass,
                          box_length=box_length)
        parts.ids = ids

        return parts

    # particles: mass, radius, ...

    def getPartsMass(self, partType, **kwargs):
        """ Get the mass (sum) of particles; an empty selection sums to 0. """
        parts = self.getParts(partType, **kwargs)
        if parts['count'] > 0:
            parts_mass = np.sum(parts['Masses'])
        else:
            parts_mass = 0.
        return parts_mass

    def getAllPartsMass(self, within_ckpch=np.inf):
        """ Get (enclosed) mass of all types of particles. """
        data = self.getAllParts(within_ckpch=within_ckpch, flatten=False)
        mass = 0.
        for partType in _ALL_PART_TYPES:
            if data[partType]['count'] > 0:
                mass += np.sum(data[partType]['Masses'])
        return mass

    def getBaryonMass(self, within_ckpch=np.inf):
        """ Get the total mass of baryon (i.e. stars and gas). """
        return (self.getPartsMass('gas', within_ckpch=within_ckpch)
                + self.getPartsMass('stars', within_ckpch=within_ckpch))

    def getPartsContainCertainMassFracRadius(
            self, partType, target_mass_frac, **kwargs):
        """ Get radius [ckpc/h] containing a specific target mass fraction. """
        parts = self.getParts(partType, **kwargs)
        if parts['count'] > 0:
            particles = self.getPartsInstance(parts)
            radius = particles.getContainCertainMassFracRadius(
                target_mass_frac=target_mass_frac)
        else:
            radius = np.nan
        return radius

    def getPartsHalfMassRadius(self, partType, **kwargs):
        """ Get the half mass radius [ckpc/h] of particles. """
        return self.getPartsContainCertainMassFracRadius(
            partType, target_mass_frac=0.5, **kwargs)

    # particles: stars properties

    def getRealStars(self, **kwargs):
        """ Get the real stars (with their birth time, photometrics, ...).
        (The birth time is in every snapshot; the photometrics and the
        luminosities are in the full snapshots only, see inFullSnap().)
        """
        # only real stars
        if 'partType' in kwargs:
            warnings.warn('partType is ignored: this method only uses real '
                          'stars.')
            del kwargs['partType']
        # only a False here is overridden; a True asks for the default
        if not kwargs.get('onlyRealStars', True):
            warnings.warn('onlyRealStars=False is overridden: this method '
                          'always uses real stars only.')
        kwargs['onlyRealStars'] = True
        # basic
        if self.inFullSnap():
            kwargs['fields'] = list(kwargs.get('fields', [])) \
                + ['GFM_StellarPhotometrics']
        cosmo = self.cosmology
        stars = self.getParts(partType='stars', **kwargs)
        # stars: birth time
        # astropy's lookback_time() refuses an empty array
        if stars['count'] > 0:
            # the scale factor at formation, the same kind of "Time" as
            # header['Time'] at the snapshot
            stars['BirthTime'] = stars['GFM_StellarFormationTime'].copy()
            stars['BirthRedshift'] = (1. / stars['BirthTime']) - 1.
            stars['BirthLookbackTime'] = cosmo.lookback_time(
                stars['BirthRedshift']).value  # Gyr
        else:
            for key in ('BirthTime', 'BirthRedshift', 'BirthLookbackTime'):
                stars[key] = np.array([], dtype=float)
        # stars: stellar photometrics and luminosities
        if self.inFullSnap():
            for idx, band in enumerate(list('UBVKgriz')):
                magnitude = stars['GFM_StellarPhotometrics'][:, idx]  # mag
                stars[f'GFM_StellarPhotometrics_{band}'] = magnitude
                stars[f'StellarLuminosities_{band}'] = \
                    absoluteMagnitudeToLuminosity(magnitude, band)
        return stars

    def getRealStarsInBirthLookbackTimeRange(
            self, birth_lbtime_range=(0., np.inf), **kwargs):
        """ Get the stars whose birth lookback time falls in a range.
        (A lookback time [Gyr] is cosmo.lookback_time(redshift).value, counted
        from z=0. The range is inclusive, and it may be given in either order.)
        """
        stars = self.getRealStars(**kwargs)
        min_lbtime = np.min(birth_lbtime_range)
        max_lbtime = np.max(birth_lbtime_range)
        cond = ((stars['BirthLookbackTime'] >= min_lbtime)
                & (stars['BirthLookbackTime'] <= max_lbtime))
        _selectParts(stars, cond)
        return stars

    def getNewlyBornStars(self, age_under, **kwargs):
        """ Get the stars younger than age_under [Gyr] at this snapshot.
        (The age runs from birth to this snapshot, not to z=0: a star younger
        than 0.1 Gyr formed no more than 0.1 Gyr before the snapshot.)
        """
        if age_under < 0.:
            raise ValueError(
                f'age_under must not be negative, not {age_under!r}')
        # the lookback time of this snapshot, not of z=0
        now = self.getLookbackTime()
        stars = self.getRealStarsInBirthLookbackTimeRange(
            birth_lbtime_range=[now, now + age_under], **kwargs)
        return stars

    def getStarsBornAtSnapshot(self, target_snapnum, **kwargs):
        """ Get stars born at a given snapshot n
        (i.e. born between snapshot n-1 and snapshot n).
        """
        if not isinstance(target_snapnum, (int, np.integer)):
            raise TypeError(
                f'snapnum must be an integer, not {target_snapnum!r}')
        if not MIN_SNAPNUM <= target_snapnum <= MAX_SNAPNUM:
            raise ValueError(
                f'snapnum {target_snapnum} is outside '
                f'{MIN_SNAPNUM}-{MAX_SNAPNUM}')
        min_lbtime = getSnapLookbackTime(self.simName, target_snapnum)
        if target_snapnum == MIN_SNAPNUM:
            # nothing is older than the first snapshot
            max_lbtime = np.inf
        else:
            max_lbtime = getSnapLookbackTime(
                self.simName, target_snapnum - 1)
        stars = self.getRealStarsInBirthLookbackTimeRange(
            birth_lbtime_range=[min_lbtime, max_lbtime], **kwargs)
        return stars

    # particles: gas properties

    def getGasContainCertainSFRFracRadius(self, target_sfr_frac, **kwargs):
        """ Get radius [ckpc/h] containing a specific fraction of SFR. """
        # only star-forming gas
        if 'partType' in kwargs:
            warnings.warn('partType is ignored: this method only uses '
                          'star-forming gas.')
            del kwargs['partType']
        if not kwargs.get('onlySFingGas', True):
            warnings.warn('onlySFingGas=False is overridden: this method only '
                          'uses star-forming gas.')
        kwargs.update({'partType': 'gas', 'onlySFingGas': True})
        parts = self.getParts(**kwargs)
        if parts['count'] == 0:
            return np.nan
        particles = self.getPartsInstance(parts)
        return getContainCertainFractionRadius(
            particles.re_dist, parts['StarFormationRate'], target_sfr_frac)

    def getGasHalfSFRRadius(self, **kwargs):
        """ Get the half star-formation rate radius [ckpc/h]. """
        return self.getGasContainCertainSFRFracRadius(
            target_sfr_frac=0.5, **kwargs)

    def getGasCoolingTime(
            self, mass_weighted_average=True, convention='zinger20',
            hydrogen_massfrac=0.76, **kwargs):
        """ Get the cooling time (Gyr) for (non-star-forming) gas cells.
        (The definition of 'zinger20' is Zinger et al. 2020
        doi:10.1093/mnras/staa2607, and 'nelson' is the n_H^2 recipe of the
        TNG forum topic 550, see tools.getCoolingTime. hydrogen_massfrac=None
        takes the fraction of every cell from GFM_Metals. With
        mass_weighted_average=False, the returned dict holds the time in its
        CoolingTime key.)
        """
        if convention not in COOLING_TIME_CONVENTIONS:
            raise ValueError(
                f'convention must be one of {COOLING_TIME_CONVENTIONS}, '
                f'not {convention!r}')
        if not self.inFullSnap():
            raise ValueError(
                'GFM_CoolingRate exists in the full snapshots only, '
                'see inFullSnap()')
        # only non-star-forming gas
        if 'partType' in kwargs:
            warnings.warn('partType is ignored: this method only uses '
                          'non-star-forming gas.')
            del kwargs['partType']
        if kwargs.get('onlySFingGas'):
            warnings.warn('onlySFingGas=True is overridden: this method only '
                          'uses non-star-forming gas.')
        kwargs['onlySFingGas'] = False
        # gas
        fields = ['InternalEnergy', 'Density', 'GFM_CoolingRate']
        # the hydrogen mass fraction is used by the 'nelson' recipe only
        if convention == 'nelson' and hydrogen_massfrac is None:
            fields.append('GFM_Metals')
        kwargs['fields'] = list(kwargs.get('fields', [])) + fields
        parts = self.getParts(partType='gas', **kwargs)
        cooling_time = np.full(parts['count'], np.nan, dtype=float)
        if parts['count'] > 0:
            if convention == 'nelson' and hydrogen_massfrac is None:
                hydrogen_massfrac = parts['GFM_Metals'][:, 0]
            rho = (parts['Density'] * massUnit(self.cosmology.h)
                   / lengthUnit(self.cosmology.h, self.header['Time'])**3)
            cooling_time = getCoolingTime(
                parts['InternalEnergy'] * (units.km / units.s)**2,
                rho,
                parts['GFM_CoolingRate']
                * (units.erg * units.cm**3 / units.s),
                hydrogen_massfrac=hydrogen_massfrac,
                convention=convention).to_value(units.Gyr)
            # the time is defined for the non-star-forming gas only
            cooling_time[parts['StarFormationRate'] > 0.] = np.nan
        parts['CoolingTime'] = cooling_time
        if mass_weighted_average:
            cond = parts['CoolingTime'] > 0.
            if not np.any(cond):
                # no non-star-forming gas with net cooling: no time to give
                return np.nan
            return np.sum(
                parts['Masses'][cond] * parts['CoolingTime'][cond]
                / np.sum(parts['Masses'][cond]))
        else:
            return parts

    def getGasTemperature(self, sfgas_temp=np.nan, **kwargs):
        """
        Get the temperature in [K] for (non-star-forming) gas cells.
        (For details refer to https://www.tng-project.org/data/docs/faq/#gen6)
        For star-forming gas cells, the temperature can be set to e.g. 10^3 K,
        or just NaN.

        References:
        (1) Volker Springel & Lars Hernquist 2003:
        '... typically assume a temperature of 10^3 K for the cold clouds ...'

        (2) Zinger et al. 2020:
        'Figure 3. ... The averages of the temperature, entropy and
        cooling time only include gas that is non-star forming and
        are mass weighted averages over the simulation slice ...'
        """
        # only non-star-forming gas
        if 'partType' in kwargs:
            warnings.warn('partType is ignored: this method only uses '
                          'non-star-forming gas.')
            del kwargs['partType']
        if kwargs.get('onlySFingGas'):
            warnings.warn('onlySFingGas=True is overridden: this method only '
                          'uses non-star-forming gas.')
        kwargs['onlySFingGas'] = False
        # gas
        kwargs['fields'] = list(kwargs.get('fields', [])) \
            + ['InternalEnergy', 'ElectronAbundance']
        parts = self.getParts(partType='gas', **kwargs)
        parts['Temperature'] = np.full(parts['count'], np.nan, dtype=float)
        if parts['count'] > 0:
            # non-star-forming gas temperature
            cond = (parts['StarFormationRate'] == 0.)  # noqa: RUF069
            parts['Temperature'][cond] = internalEnergyToTemperature(
                parts['InternalEnergy'][cond],
                parts['ElectronAbundance'][cond])
            # star-forming gas temperature
            parts['Temperature'][~cond] = sfgas_temp
        return parts

    def getGasHydrogenNumberDensity(self, **kwargs):
        """ Get the hydrogen number density [cm^-3] for non-star-forming gas.
        (The hydrogen mass fraction is GFM_Metals[:, 0] in the full snapshot
        and 0.76, the primordial value, in the mini snapshots.)
        """
        # only non-star-forming gas
        if 'partType' in kwargs:
            warnings.warn('partType is ignored: this method only uses '
                          'non-star-forming gas.')
            del kwargs['partType']
        if kwargs.get('onlySFingGas'):
            warnings.warn('onlySFingGas=True is overridden: this method only '
                          'uses non-star-forming gas.')
        kwargs['onlySFingGas'] = False
        # gas
        fields = ['Density']
        if self.inFullSnap():
            fields.append('GFM_Metals')
        kwargs['fields'] = list(kwargs.get('fields', [])) + fields
        parts = self.getParts(partType='gas', **kwargs)
        # non-star-forming gas hydrogen number density: nH
        parts['HydrogenNumberDensity'] = np.full(
            parts['count'], np.nan, dtype=float)
        if parts['count'] > 0:
            cosmo = self.cosmology
            header = self.header
            cond = (parts['StarFormationRate'] == 0.)  # noqa: RUF069
            if self.inFullSnap():
                X_H = parts['GFM_Metals'][:, 0]
            else:
                warnings.warn(
                    'GFM_Metals exists in the full snapshots only, so the '
                    'hydrogen mass fraction is taken as 0.76.', stacklevel=2)
                X_H = 0.76
            rho = (parts['Density'] * massUnit(cosmo.h)
                   / lengthUnit(cosmo.h, header['Time'])**3)
            m_p = const.m_p  # pyright: ignore
            n_H = (X_H * rho / m_p).to(units.cm**-3)
            parts['HydrogenNumberDensity'][cond] = n_H.value[cond]
        return parts

    # kinematic

    def getAngularMomentum(self, partType, specific=True, withUnit=True,
                           norm=False, center='object', **kwargs):
        """ Get the (specific) angular momentum about a center.
        (center='object' measures about the minimum of the potential, the
        SubhaloPos or the GroupPos: what the spin parameter, the misalignment
        with the host and the tidal torques of a halo need. center='parts'
        measures about the mass weighted center of the particles themselves
        and their mass weighted mean velocity: the spin of the galaxy, and
        what the SpecificAngMom of the TNG supplementary catalog reports. The
        first is the second plus M dr x dv, the orbital angular momentum of
        the particle center about the potential center, where dr and dv are
        the offsets of that center and of its velocity. The radius of
        within_ckpch is always measured about the center of the object,
        whichever center the angular momentum itself uses.)
        """
        if center not in _ALLOWED_CENTERS:
            raise ValueError(
                f'center must be one of {_ALLOWED_CENTERS}, not {center!r}')
        if specific:
            default_unit = units.kpc * units.km / units.s
        else:
            default_unit = units.Msun * units.kpc * units.km / units.s

        # calculate the angular momentum
        parts = self.getParts(partType, **kwargs)
        if parts['count'] > 0:
            particles = self.getPartsInstance(
                parts, with_physical_units=True, center=center)
            ang_mom = particles.getAngularMomentum(
                specific=specific, stack=True)
            ang_mom = ang_mom.to(unit=default_unit)  # pyright: ignore
        else:
            ang_mom = np.full(3, np.nan, dtype=float) * default_unit

        if not withUnit:
            ang_mom = ang_mom.to_value(unit=default_unit)  # pyright: ignore

        if norm:
            ang_mom = np.linalg.norm(ang_mom)

        return ang_mom

    def getAllPartsAngularMomentum(
            self, within_ckpch=np.inf, specific=True, norm=False,
            withUnit=True, center='object'):
        """ Get the total (specific) angular momentum of all the types.
        (Every type goes into one system, so center='parts' takes the mass
        weighted center of all the particles together. That center can sit
        away from the minimum of the potential, but for the whole subhalo the
        two give the same answer: SubhaloVel already is the mass weighted
        mean velocity of these same particles, so the velocity offset of the
        two centers is zero. A subset, the stars alone or a radius cut, has a
        mean velocity of its own, and there the two centers differ. The
        radius of within_ckpch is measured about the center of the object.)
        """
        if center not in _ALLOWED_CENTERS:
            raise ValueError(
                f'center must be one of {_ALLOWED_CENTERS}, not {center!r}')
        default_unit = units.Msun * units.kpc * units.km / units.s
        if specific:
            default_unit = units.kpc * units.km / units.s
        all_parts = self.getAllParts(within_ckpch=within_ckpch, flatten=True)
        if all_parts['count'] == 0:
            # no particle inside the radius: no angular momentum to report
            ang_mom = np.full(3, np.nan) * default_unit
        else:
            particles = self.getPartsInstance(
                all_parts, with_physical_units=True, center=center)
            ang_mom = particles.getAngularMomentum(
                specific=specific, stack=True)
            ang_mom = ang_mom.to(unit=default_unit)  # pyright: ignore
        if not withUnit:
            ang_mom = ang_mom.to_value(unit=default_unit)  # pyright: ignore
        if norm:
            ang_mom = np.linalg.norm(ang_mom)
        return ang_mom
