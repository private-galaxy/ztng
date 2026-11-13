"""
Runnable checks of ztng.classes.particles.

Usage:
    python tests/test_particles.py
    pytest tests/test_particles.py

Every check builds a small set of particles with a known answer, so it needs
neither data files nor a network.
"""

import os
import sys
import warnings

import numpy as np
from astropy import constants as const
from astropy import units

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ztng.classes.particles import Particles  # noqa: E402


def _particles(*args, **kwargs):
    """Build a Particles instance that does not cross the box edge."""
    kwargs.setdefault('box_length', np.inf)
    return Particles(*args, **kwargs)


def ring(count=16, radius=10.):
    """Return coordinates and circular velocities on a ring."""
    angle = np.linspace(0., 2. * np.pi, count, endpoint=False)
    coordinates = np.c_[radius * np.cos(angle),
                        radius * np.sin(angle), np.zeros(count)]
    velocities = np.c_[-np.sin(angle), np.cos(angle), np.zeros(count)]
    return coordinates, velocities


def test_centre_is_the_mass_weighted_average():
    """The centre sits at the mass weighted average of the coordinates."""
    parts = _particles(
        masses=np.array([1., 3.]),
        coordinates=np.array([[0., 0., 0.], [4., 0., 0.]]),
        velocities=np.zeros((2, 3)))
    assert parts.count == 2
    assert np.allclose(parts.center_position, [3., 0., 0.])
    assert np.allclose(parts.re_coor[:, 0], [-3., 1.])
    assert np.allclose(parts.re_dist, [3., 1.])


def test_given_centre_is_kept():
    """A given centre is used as it is, without averaging."""
    parts = _particles(
        masses=np.array([1., 3.]),
        coordinates=np.array([[0., 0., 0.], [4., 0., 0.]]),
        velocities=np.zeros((2, 3)),
        center_position=np.array([2., 0., 0.]),
        center_velocity=np.zeros(3))
    assert np.allclose(parts.center_position, [2., 0., 0.])
    assert np.allclose(parts.re_coor[:, 0], [-2., 2.])


def test_box_length_wraps_the_relative_position():
    """A box length keeps a system across the box edge at its true
    distance, and a zero distance stays zero."""
    coordinates = np.array([[0.1, 0., 0.], [999.9, 0., 0.]])
    velocities = np.zeros((2, 3))
    centre = np.zeros(3)
    plain = _particles(np.ones(2), coordinates, velocities,
                       center_position=centre, center_velocity=centre)
    assert np.allclose(plain.re_dist, [0.1, 999.9]), plain.re_dist
    wrapped = _particles(np.ones(2), coordinates, velocities,
                         center_position=centre, center_velocity=centre,
                         box_length=1000.)
    assert np.allclose(wrapped.re_dist, [0.1, 0.1]), wrapped.re_dist
    assert np.allclose(wrapped.re_coor[:, 0], [0.1, -0.1])


def test_missing_box_length_warns():
    """A Particles instance without a box length says what may go wrong."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        parts = Particles(np.ones(2), np.zeros((2, 3)), np.zeros((2, 3)))
    assert any('box_length' in str(item.message) for item in caught), caught
    assert np.allclose(parts.re_dist, 0.)


def test_an_infinite_box_length_does_not_wrap():
    """np.inf says that the system does not cross the box edge."""
    coordinates = np.array([[900., 0., 0.], [-900., 0., 0.]])
    parts = _particles(np.ones(2), coordinates, np.zeros((2, 3)),
                       center_position=np.zeros(3),
                       center_velocity=np.zeros(3), box_length=np.inf)
    assert np.allclose(parts.re_dist, [900., 900.]), parts.re_dist


def test_lists_are_accepted():
    """Lists become arrays, so every method works on them too."""
    parts = _particles([1., 2.], [[1., 0., 0.], [-1., 0., 0.]],
                       [[0., 1., 0.], [0., -1., 0.]],
                       center_position=[0., 0., 0.],
                       center_velocity=[0., 0., 0.])
    assert isinstance(parts.masses, np.ndarray)
    assert np.allclose(parts.getAngularMomentum(specific=False, stack=True),
                       [0., 0., 3.])


def test_arrays_of_different_lengths_are_refused():
    """The three arrays must describe the same particles."""
    good = (np.ones(3), np.zeros((3, 3)), np.zeros((3, 3)))
    for index in (1, 2):
        bad = list(good)
        bad[index] = bad[index][:2]
        try:
            _particles(*bad)
        except ValueError as error:
            assert 'same length' in str(error)
        else:
            raise AssertionError(f'index {index} should raise ValueError')


def test_no_particle_is_refused():
    """An empty instance has no centre and no half mass radius."""
    try:
        _particles(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)))
    except ValueError as error:
        assert 'at least one particle' in str(error)
    else:
        raise AssertionError('An empty instance should raise ValueError')


def test_half_mass_radius_of_a_uniform_sphere():
    """A uniform sphere holds half of its mass within R * 0.5 ** (1 / 3)."""
    rng = np.random.default_rng(0)
    count, radius = 40000, 10.
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, np.newaxis]
    # A uniform density gives r = radius * u ** (1 / 3) for uniform u.
    r = radius * rng.uniform(0., 1., count) ** (1. / 3.)
    parts = _particles(np.ones(count), direction * r[:, np.newaxis],
                       np.zeros((count, 3)))
    assert np.isclose(parts.getHalfMassRadius(),
                      radius * 0.5 ** (1. / 3.), rtol=0.01)
    # An eighth of the mass fills half of the radius.
    eighth = count / 8.
    assert np.isclose(parts.getContainCertainMassRadius(eighth),
                      radius / 2., rtol=0.02)


def test_angular_momentum_of_four_combinations():
    """The four combinations agree with the values by hand."""
    parts = _particles(
        masses=np.array([1., 2.]),
        coordinates=np.array([[1., 0., 0.], [-1., 0., 0.]]),
        velocities=np.array([[0., 1., 0.], [0., -1., 0.]]),
        center_position=np.zeros(3),
        center_velocity=np.zeros(3))
    # The cross product is (0, 0, 1) for both particles.
    assert np.allclose(
        parts.getAngularMomentum(specific=True, stack=False),
        [[0., 0., 1.], [0., 0., 1.]])
    assert np.allclose(
        parts.getAngularMomentum(specific=False, stack=False),
        [[0., 0., 1.], [0., 0., 2.]])
    assert np.allclose(
        parts.getAngularMomentum(specific=False, stack=True), [0., 0., 3.])
    assert np.allclose(
        parts.getAngularMomentum(specific=True, stack=True), [0., 0., 1.])


def test_jz_follows_the_total_angular_momentum():
    """Three particles turn one way and one turns the other way."""
    parts = _particles(
        masses=np.ones(4),
        coordinates=np.array([[0., 1., 0.], [0., 2., 0.], [0., 3., 0.],
                              [0., -1., 0.]]),
        velocities=np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.],
                             [1., 0., 0.]]),
        center_position=np.zeros(3),
        center_velocity=np.zeros(3))
    assert np.allclose(parts.getPerParticleJz(), [1., 2., 3., -1.])
    assert np.allclose(parts.getPerParticleJzSign(), [1., 1., 1., -1.])
    # One particle out of four turns the other way, counted twice.
    assert np.isclose(parts.getKinematicBulgeFraction(), 0.5)


def test_circularity_of_a_circular_orbit():
    """A circular orbit has |jz| = j_circ, so the circularity is 1."""
    mass = 1e10 * units.Msun
    radius = 10. * units.kpc
    speed = np.sqrt(const.G * mass / radius).to(units.km / units.s)
    parts = _particles(
        masses=np.array([1.]) * mass,
        coordinates=np.array([[1., 0., 0.]]) * radius,
        velocities=np.array([[0., 1., 0.]]) * speed,
        center_position=np.zeros(3) * radius,
        center_velocity=np.zeros(3) * units.km / units.s,
        enclosed_mass=np.array([1.]) * mass)
    assert np.isclose(parts.getCircularities()[0], 1., rtol=1e-6)
    # The half of the particles above 0.7 is all of them.
    assert np.isclose(parts.getKinematicDiscFraction(0.7), 1.)


def test_circularity_needs_the_enclosed_mass():
    """The circularity cannot be computed from the particles alone."""
    coordinates, velocities = ring(4)
    parts = _particles(np.ones(4), coordinates, velocities,
                       center_position=np.zeros(3),
                       center_velocity=np.zeros(3))
    for call in (parts.getCircularities, parts.getKinematicDiscFraction):
        try:
            call()
        except ValueError as error:
            assert 'enclosed' in str(error)
        else:
            raise AssertionError(f'{call.__name__} should raise ValueError')


def test_circularity_needs_physical_units():
    """Without units the circularity would carry a meaningless scale."""
    parts = _particles(np.ones(3), np.eye(3), np.eye(3),
                       center_position=np.zeros(3),
                       center_velocity=np.zeros(3),
                       enclosed_mass=np.ones(3))
    try:
        parts.getCircularities()
    except TypeError as error:
        assert 'physical units' in str(error)
    else:
        raise AssertionError('Unitless input should raise TypeError')


def test_kappa_of_a_rigid_rotation():
    """Circular motion puts all the kinetic energy into rotation."""
    coordinates, velocities = ring()
    parts = _particles(np.ones(16), coordinates, velocities,
                       center_position=np.zeros(3),
                       center_velocity=np.zeros(3))
    assert np.isclose(parts.getKinematicKappa(), 1.)
    # The same ring moving radially has no angular momentum at all.
    radial = coordinates / np.linalg.norm(coordinates, axis=1)[:, np.newaxis]
    parts = _particles(np.ones(16), coordinates, radial,
                       center_position=np.zeros(3),
                       center_velocity=np.zeros(3))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert np.isclose(parts.getKinematicKappa(), 0.)


def test_kappa_without_kinetic_energy():
    """A system at rest has no total kinetic energy, so kappa is NaN."""
    parts = _particles(np.ones(3), np.eye(3), np.zeros((3, 3)),
                       center_position=np.zeros(3),
                       center_velocity=np.zeros(3))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert np.isnan(parts.getKinematicKappa())


def test_zero_total_angular_momentum_warns():
    """Two opposite orbits cancel, so jz has no direction."""
    parts = _particles(
        masses=np.ones(2),
        coordinates=np.array([[1., 0., 0.], [-1., 0., 0.]]),
        velocities=np.array([[0., 1., 0.], [0., 1., 0.]]),
        center_position=np.zeros(3),
        center_velocity=np.zeros(3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        per_part_jz = parts.getPerParticleJz()
        bulge = parts.getKinematicBulgeFraction()
    assert 'total angular momentum is zero' in str(caught[0].message)
    assert np.all(np.isnan(per_part_jz))
    # An undefined fraction stays NaN instead of falling back to zero.
    assert np.isnan(bulge)


def test_rotation_matrix_of_a_face_on_view():
    """The matrix is a rotation that puts the angular momentum on z."""
    parts = _particles(
        masses=np.ones(1),
        coordinates=np.array([[1., 0., 0.]]),
        velocities=np.array([[0., 0., 1.]]),
        center_position=np.zeros(3),
        center_velocity=np.zeros(3))
    matrix = parts.getRotationTransformationMatrix()
    assert np.isclose(np.linalg.det(matrix), 1.)
    assert np.allclose(np.dot(matrix, matrix.T), np.eye(3))
    ang_mom = parts.getAngularMomentum()
    unit_ang_mom = ang_mom / np.linalg.norm(ang_mom)
    assert np.allclose(np.dot(unit_ang_mom, matrix), [0., 0., 1.])


if __name__ == '__main__':
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith('test_') and callable(obj)]
    for check in checks:
        check()
        print(f'PASS {check.__name__}')
    print(f'{len(checks)} checks passed')
