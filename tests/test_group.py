"""
Runnable checks of ztng.classes.group.

Usage:
    python tests/test_group.py
    pytest tests/test_group.py

Every check works on four stars held in memory, so it needs neither data files
nor a network. The fake group object replaces the snapshot reader and the two
lookups that would leave this machine.
"""

import os
import sys
import warnings

import numpy as np
from astropy.cosmology import FlatLambdaCDM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ztng.classes import group as groupModule  # noqa: E402
from ztng.classes.group import Group  # noqa: E402

# getStarsBornAtSnapshot() reads the lookback time through the module-level
# function instead of through self, so the fake value has to be installed
# here.
groupModule.getSnapLookbackTime = lambda simName, snapnum: 13. - 0.1 * snapnum


class fakeGroup(Group):
    """ A group object that serves four stars from memory. """

    def __init__(self, snapnum=99):
        super().__init__('fake', snapnum)
        self.center_position = np.zeros(3)
        self.center_velocity = np.zeros(3)

    def getCosmology(self):
        return FlatLambdaCDM(H0=70., Om0=0.3)

    def getHeader(self):
        return {'Time': 0.9}

    def getLookbackTime(self):
        return 13.

    def getSimMeta(self):
        return {'mass_dm': 0.25, 'boxsize': 1000.}

    def getSnap(self, partType, fields):
        data = {
            'count': 4,
            'ParticleIDs': np.arange(4),
            'Masses': np.full(4, 0.25),
            'Coordinates': np.array([
                [1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [-1., 0., 0.]]),
            'Velocities': np.zeros((4, 3)),
        }
        if 'GFM_StellarFormationTime' in fields:
            # scale factors that give 1.3, 7.9, 10.4 and 11.7 Gyr of lookback
            data['GFM_StellarFormationTime'] = np.array([0.9, 0.5, 0.3, 0.2])
        if 'GFM_StellarPhotometrics' in fields:
            data['GFM_StellarPhotometrics'] = np.zeros((4, 8))
        for name in ('StarFormationRate', 'InternalEnergy', 'Density',
                     'ElectronAbundance'):
            if name in fields:
                data[name] = np.ones(4)
        if 'GFM_CoolingRate' in fields:
            data['GFM_CoolingRate'] = -np.ones(4)
        if 'GFM_Metals' in fields:
            data['GFM_Metals'] = np.zeros((4, 11))
        return data


class spinningGroup(fakeGroup):
    """ Four stars that move, so their angular momentum is not zero. """

    def getSnap(self, partType, fields):
        data = super().getSnap(partType, fields)
        data['Velocities'] = np.array([
            [0., 10., 0.], [0., 0., 10.], [10., 0., 0.], [0., 0., 0.]])
        return data


class edgeGroup(fakeGroup):
    """ Two stars at both ends of the box, so the edge is their mean. """

    def getSimMeta(self):
        return {'mass_dm': 0.25, 'boxsize': 1000.}

    def getSnap(self, partType, fields):
        data = super().getSnap(partType, fields)
        data['Coordinates'] = np.array([
            [0.1, 0., 0.], [999.9, 0., 0.], [0.1, 0., 0.], [999.9, 0., 0.]])
        data['Velocities'] = np.array([
            [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]])
        return data


def test_the_center_of_the_parts_wraps_at_the_box_edge():
    """Two stars at the two ends of the box have their center at the edge,
    not in the middle of the box."""
    group = edgeGroup()
    physics = group.getPartsInstance(
        group.getParts('stars'), with_physical_units=False, center='parts')
    center = np.asarray(physics.center_position, dtype=float)
    assert np.allclose(center, [0., 0., 0.]), center
    assert np.allclose(np.asarray(physics.re_coor, dtype=float)[:, 0],
                       [0.1, -0.1, 0.1, -0.1])


def test_in_full_snap():
    """ Only the snapshots in the TNG full list report as full. """
    for snapnum in (99, 50, 2):
        assert fakeGroup(snapnum).inFullSnap(), snapnum
    for snapnum in (98, 51, 0):
        assert not fakeGroup(snapnum).inFullSnap(), snapnum


def test_the_center_of_the_parts_is_their_mass_weighted_center():
    """center='parts' measures about the particles, as Particles does alone."""
    from ztng.classes.particles import Particles
    group = spinningGroup()
    parts = group.getParts('stars')
    physics = group.getPartsInstance(parts, with_physical_units=True)
    # the same particles about their own center, built by hand
    direct = Particles(physics.masses, physics.coordinates,
                       physics.velocities, box_length=np.inf)
    want = direct.getAngularMomentum(specific=True, stack=True)
    got = group.getAngularMomentum(
        partType='stars', specific=True, withUnit=False, center='parts')
    assert np.allclose(got, want.value, rtol=1e-12), (got, want)
    # the center of the object is a different answer for these stars
    about_object = group.getAngularMomentum(
        partType='stars', specific=True, withUnit=False)
    assert not np.allclose(about_object, got), (about_object, got)


def test_all_parts_without_a_particle_is_nan():
    """A radius that holds nothing gives NaN, not a division by zero."""
    group = spinningGroup()
    got = group.getAllPartsAngularMomentum(within_ckpch=0.)
    assert np.all(np.isnan(got.value)), got
    assert str(got.unit) == 'km kpc / s', got.unit
    plain = group.getAllPartsAngularMomentum(within_ckpch=0.,
                                             withUnit=False)
    assert np.all(np.isnan(plain)), plain


def test_all_parts_about_the_center_of_all_the_particles():
    """center='parts' takes one center for every type together."""
    group = spinningGroup()
    got = group.getAllPartsAngularMomentum(center='parts', withUnit=False)
    physics = group.getPartsInstance(
        group.getAllParts(flatten=True), with_physical_units=True,
        center='parts')
    want = physics.getAngularMomentum(specific=True, stack=True)
    assert np.allclose(got, want.value, rtol=1e-12), (got, want)


def test_an_unknown_center_is_an_error():
    """A center that is not 'object' or 'parts' must not pass in silence."""
    group = spinningGroup()
    for call in (
            lambda: group.getPartsInstance(
                group.getParts('stars'), center='com'),
            lambda: group.getAngularMomentum('stars', center='com'),
            lambda: group.getAllPartsAngularMomentum(center='com'),
            # an empty selection must not swallow the bad value either
            lambda: group.getAllPartsAngularMomentum(
                within_ckpch=0., center='com')):
        try:
            call()
        except ValueError as err:
            assert 'center must be' in str(err), err
        else:
            raise AssertionError('an unknown center did not raise')


def test_parts_returns_a_new_dict():
    """Two calls with the same arguments must not hand out the same dict."""
    group = fakeGroup()
    first = group.getParts('stars')
    second = group.getParts('stars')
    assert first is not second
    assert first.keys() == second.keys()


def test_adding_a_key_does_not_touch_the_cache():
    """A key added to the result must not appear in the next call."""
    group = fakeGroup()
    group.getParts('stars')['InjectedByCaller'] = 1
    assert 'InjectedByCaller' not in group.getParts('stars')


def test_replacing_a_key_does_not_touch_the_cache():
    """A key replaced in the result must keep its value in the next call."""
    group = fakeGroup()
    parts = group.getParts('stars')
    parts['Masses'] = parts['Masses'][:1]
    parts['count'] = 1
    whole = group.getParts('stars')
    assert whole['count'] == 4
    assert len(whole['Masses']) == 4


def test_a_flag_for_another_type_warns():
    """A flag that cannot apply to the type must not pass in silence."""
    group = fakeGroup()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        group.getParts('dm', onlyRealStars=True)
        group.getParts('stars', onlySFingGas=True)
    messages = [str(warning.message) for warning in caught]
    assert len(messages) == 2, messages
    assert 'onlyRealStars' in messages[0], messages
    assert 'onlySFingGas' in messages[1], messages
    # A flag for its own type stays quiet.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        group.getParts('stars', onlyRealStars=True)
        group.getParts('gas', onlySFingGas=True)
    assert not caught, [str(warning.message) for warning in caught]


def test_birth_time_filter_keeps_the_cache():
    """A birth-time filter must not shrink what the next call returns."""
    group = fakeGroup()
    count_before = group.getRealStars()['count']
    assert count_before == 4
    filtered = group.getRealStarsInBirthLookbackTimeRange(
        birth_lbtime_range=[0., 2.])
    assert filtered['count'] == 1
    assert group.getRealStars()['count'] == count_before


def test_newborn_stars_keep_the_cache():
    """The same, through the method that wraps the filter."""
    group = fakeGroup()
    count_before = group.getRealStars()['count']
    assert group.getNewlyBornStars(age_under=0.5)['count'] < count_before
    assert group.getRealStars()['count'] == count_before


def test_derived_keys_do_not_leak_into_the_cache():
    """Keys added by a derived method must not show up in a plain call."""
    group = fakeGroup()
    stars = group.getRealStars()
    assert 'BirthTime' in stars
    assert 'StellarLuminosities_U' in stars
    bare = group.getParts(
        'stars', onlyRealStars=True, fields=['GFM_StellarPhotometrics'])
    assert 'BirthTime' not in bare
    assert 'StellarLuminosities_U' not in bare


def test_snapnum_out_of_range_is_an_error():
    """A snapshot number that does not exist is an error, not an empty set."""
    group = fakeGroup()
    for snapnum in (-1, 100, 1000):
        try:
            group.getStarsBornAtSnapshot(target_snapnum=snapnum)
        except ValueError as err:
            assert str(snapnum) in str(err)
        else:
            raise AssertionError(f'snapnum {snapnum} did not raise')
    stars = group.getStarsBornAtSnapshot(target_snapnum=98)
    assert stars['count'] == 0


def test_a_negative_age_is_an_error():
    """A negative age is a sign error, not an empty selection."""
    group = fakeGroup()
    for age in (-1., -0.001):
        try:
            group.getNewlyBornStars(age_under=age)
        except ValueError as err:
            assert 'must not be negative' in str(err), err
        else:
            raise AssertionError(f'age_under {age} did not raise')
    # Zero is allowed: it selects the stars born at this very moment.
    assert group.getNewlyBornStars(age_under=0.)['count'] == 0


def test_a_snapshot_that_is_not_an_integer_is_an_error():
    """Half a snapshot does not exist, and must not reach the reader."""
    group = fakeGroup()
    for snapnum in (98.5, '99', None):
        try:
            group.getStarsBornAtSnapshot(target_snapnum=snapnum)
        except TypeError as err:
            assert 'must be an integer' in str(err), err
        else:
            raise AssertionError(f'snapnum {snapnum!r} did not raise')


def test_every_result_has_a_count():
    """getParts() and getAllParts() always report how many they hold."""
    group = fakeGroup()
    for partType in ['dm', 'gas', 'stars', 'bh']:
        assert 'count' in group.getParts(partType), partType
    all_parts = group.getAllParts()
    for partType in ['dm', 'gas', 'stars', 'bh']:
        assert 'count' in all_parts[partType], partType
    assert 'count' in group.getAllParts(flatten=True)


def test_a_radius_range_wraps_across_the_box():
    """An inner radius must not catch the particles on the other side of
    the box."""
    group = fakeGroup()
    group.getSnap = lambda partType, fields: {
        'count': 3,
        'ParticleIDs': np.arange(3),
        'Masses': np.ones(3),
        'Coordinates': np.array([[0.1, 0., 0.], [999.9, 0., 0.],
                                 [5., 0., 0.]]),
        'Velocities': np.zeros((3, 3)),
    }
    # the box is 1000 ckpc/h, so 999.9 sits 0.2 away from the center
    assert group.getParts('dm', within_ckpch=[4., np.inf])['count'] == 1
    assert group.getParts('dm', within_ckpch=[0., np.inf])['count'] == 3


def test_verbose_does_not_split_the_cache():
    """Do not keep a second copy of the arrays for verbose=True."""
    group = fakeGroup()
    group.getParts('stars')
    group.getParts('stars', verbose=True)
    assert len(group._cache) == 1, len(group._cache)
    assert group.getParts('stars', verbose=True)['count'] == 4


def test_a_dropped_partType_warns():
    """A partType that a method cannot use is reported, then dropped."""
    group = fakeGroup()
    calls = (
        lambda: group.getRealStars(partType='stars'),
        lambda: group.getGasContainCertainSFRFracRadius(
            target_sfr_frac=0.5, partType='stars'),
        lambda: group.getGasTemperature(partType='stars'),
        lambda: group.getGasCoolingTime(partType='stars'),
        lambda: group.getGasHydrogenNumberDensity(partType='stars'),
    )
    for call in calls:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            call()
        messages = [str(item.message) for item in caught]
        assert any('partType is ignored' in text for text in messages)


def test_a_forced_flag_warns_when_it_conflicts():
    """A flag that a method overrides is reported when the value differs."""
    group = fakeGroup()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        group.getRealStars(onlyRealStars=False)
    messages = [str(item.message) for item in caught]
    assert any('is overridden' in text for text in messages), messages
    # A value that asks for what the method does anyway stays quiet, and so
    # does a call that leaves the flag out.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        group.getRealStars(onlyRealStars=True)
        group.getRealStars()
        group.getGasTemperature()
    assert not caught, [str(item.message) for item in caught]


def test_a_forced_gas_phase_is_reported_when_it_conflicts():
    """The gas methods report the phase they force, but only on a conflict."""
    group = fakeGroup()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        group.getGasTemperature(onlySFingGas=True)
        group.getGasContainCertainSFRFracRadius(
            target_sfr_frac=0.5, onlySFingGas=False)
    messages = [str(item.message) for item in caught]
    assert any('is overridden' in text for text in messages), messages
    # A value the method uses anyway stays quiet, and so does a call that
    # leaves the flag out.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        group.getGasTemperature(onlySFingGas=False)
        group.getGasTemperature()
        group.getGasContainCertainSFRFracRadius(
            target_sfr_frac=0.5, onlySFingGas=True)
        group.getGasContainCertainSFRFracRadius(target_sfr_frac=0.5)
    assert not caught, [str(item.message) for item in caught]


def test_a_mini_snapshot_falls_back_to_the_primordial_fraction():
    """n_H reads GFM_Metals, or takes 0.76 where that field is absent."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fakeGroup(98).getGasHydrogenNumberDensity()
    messages = [str(item.message) for item in caught]
    assert any('0.76' in text for text in messages), messages
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fakeGroup(99).getGasHydrogenNumberDensity()
    assert not caught, [str(item.message) for item in caught]


def test_the_cooling_time_needs_a_full_snapshot():
    """GFM_CoolingRate does not exist in the mini snapshots."""
    try:
        fakeGroup(98).getGasCoolingTime()
    except ValueError as err:
        assert 'full snapshots only' in str(err), err
    else:
        raise AssertionError('a mini snapshot did not raise')


def test_enclosed_mass_counts_every_type():
    """The enclosed mass sums all particle types, not only the selection."""
    group = fakeGroup()
    parts = group.getParts('stars', within_ckpch=np.inf)
    instance = group.getPartsInstance(parts, input_enclosed_mass=True)
    # Four stars sit at 1, 2, 3 and 1 ckpc/h; every type serves the same set,
    # so M(<r) is 4 x 0.25 per particle of any type that lies inside.
    assert np.allclose(instance.enclosed_mass, [0., 2., 3., 0.])


def test_cooling_time_is_nan_without_cooling_gas():
    """Gas that is all star forming has no cooling time to average."""
    group = fakeGroup()
    # The fake serves StarFormationRate = 1, so no cell is available.
    assert np.isnan(group.getGasCoolingTime())
    parts = group.getGasCoolingTime(mass_weighted_average=False)
    assert np.all(np.isnan(parts['CoolingTime']))


def test_parts_mass_is_zero_for_an_empty_selection():
    """An empty selection sums to zero, not to NaN."""
    group = fakeGroup()
    assert group.getPartsMass('stars', within_ckpch=1e-6) == 0.


def test_a_named_radius_needs_its_base():
    """A named radius without the radius it refers to is an error."""
    group = fakeGroup()
    for name in ('R200', '0d1R200', 'Re', '2Re', '10Re'):
        try:
            group.getParts('stars', within_ckpch=name)
        except ValueError as err:
            assert name in str(err)
        else:
            raise AssertionError(f'within_ckpch={name!r} did not raise')


def test_a_named_radius_needs_a_positive_base():
    """A zero or negative base radius would select nothing, so it is an
    error rather than an empty result."""
    group = fakeGroup()
    for base in (0., -1., np.nan):
        try:
            group.getParts('stars', within_ckpch='Re', Re_ckpch=base)
        except ValueError as err:
            assert 'Re_ckpch' in str(err), str(err)
        else:
            raise AssertionError(f'Re_ckpch={base!r} did not raise')
    assert group.getParts(
        'stars', within_ckpch='Re', Re_ckpch=1.)['count'] > 0


def test_a_named_radius_uses_the_base():
    """A named radius keeps the particles inside that many base radii."""
    group = fakeGroup()
    assert group.getParts(
        'stars', within_ckpch='R200', R200_ckpch=2.5)['count'] == 3
    assert group.getParts(
        'stars', within_ckpch='R200', R200_ckpch=0.5)['count'] == 0


if __name__ == '__main__':
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith('test_') and callable(obj)]
    for check in checks:
        check()
        print(f'PASS {check.__name__}')
    print(f'{len(checks)} checks passed')
