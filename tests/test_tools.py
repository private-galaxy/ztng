"""
Runnable checks of ztng.tools.

Usage:
    python tests/test_tools.py
    pytest tests/test_tools.py

The helpers work on arrays, so every check builds its own small input and
needs neither data files nor a network.
"""

import itertools
import os
import sys
import warnings

import numpy as np
from astropy import constants as const
from astropy import units

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ztng import tools  # noqa: E402


def test_divide():
    """Divide returns the default where the denominator is zero."""
    out = tools.divide(np.array([1., 2., 3.]), np.array([2., 0., 4.]))
    assert np.allclose(out[:1], 0.5)
    assert np.isnan(out[1])
    assert np.allclose(out[2:], 0.75)
    # The default can be a number instead of NaN.
    zeros = np.zeros(3)
    assert np.all(tools.divide(zeros, zeros, default_out=0.) == 0.)
    # Integer arrays work, the output is a float array.
    out = tools.divide(np.array([1, 3]), np.array([2, 0]))
    assert np.isclose(out[0], 0.5) and np.isnan(out[1])
    assert out.dtype == float
    # A tiny but non-zero denominator is kept, not treated as zero.
    assert np.isfinite(tools.divide(np.array([1.]), np.array([1e-300]))[0])
    # Negative zero is zero, and an infinite denominator gives zero.
    assert np.isnan(tools.divide(np.array([1.]), np.array([-0.]))[0])
    assert tools.divide(np.array([1.]), np.array([np.inf]))[0] == 0.


def test_log10():
    """log10 returns the default where the input is not positive."""
    out = tools.log10(np.array([100., 0., -5.]))
    assert np.allclose(out[0], 2.)
    assert np.isnan(out[1]) and np.isnan(out[2])
    # Integer arrays work, the output is a float array.
    out = tools.log10(np.array([100, 0]))
    assert np.isclose(out[0], 2.) and np.isnan(out[1])
    assert out.dtype == float
    # A scalar and a very small positive number are handled too.
    assert np.isclose(tools.log10(4.), np.log10(4.))
    assert np.isfinite(tools.log10(np.array([1e-300]))[0])


def test_angle_between():
    """The angle between two vectors matches the axes they lie on."""
    x_axis = np.array([1., 0., 0.])
    y_axis = np.array([0., 1., 0.])
    angle = tools.getAngleBetween(x_axis, y_axis)
    assert np.isclose(angle.to(units.rad).value, np.pi / 2)
    angle = tools.getAngleBetween(x_axis, y_axis, angle_unit='degree')
    assert np.isclose(angle.to(units.degree).value, 90.)
    # Parallel vectors enclose a zero angle, not a NaN.
    angle = tools.getAngleBetween(x_axis, 3. * x_axis)
    assert np.isclose(angle.to(units.rad).value, 0., atol=1e-7)
    # A Quantity input still gives an angle in rad, not in rad squared.
    vector = np.array([0., 1., 0.]) * units.kpc
    angle = tools.getAngleBetween(vector, np.array([0., 0., 1.]))
    assert angle.unit == units.rad
    assert np.isclose(angle.to(units.degree).value, 90.)
    # Opposite vectors enclose pi, and the lengths do not matter.
    assert np.isclose(tools.getAngleBetween(x_axis, -x_axis)
                      .to(units.rad).value, np.pi)
    assert np.isclose(tools.getAngleBetween(3. * x_axis, 0.5 * y_axis)
                      .to(units.rad).value, np.pi / 2)
    # An analytic 45 degrees.
    assert np.isclose(tools.getAngleBetween(np.array([1., 1., 0.]), x_axis,
                                            'degree').to(units.degree).value,
                      45.)
    # A zero vector has no direction, so the angle is NaN.
    assert np.isnan(tools.getAngleBetween(np.zeros(3), x_axis).value)
    # 100 random pairs against arccos of the normalized dot product.
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, 100, 3))
    want = np.arccos(np.clip(
        np.sum(a * b, -1) / np.linalg.norm(a, axis=1)
        / np.linalg.norm(b, axis=1), -1, 1))
    got = np.array([tools.getAngleBetween(a[idx], b[idx]).to(units.rad).value
                    for idx in range(100)])
    assert np.allclose(got, want)


def test_angle_between_rejects_unknown_unit():
    """A misspelled unit raises instead of returning radians."""
    x_axis = np.array([1., 0., 0.])
    for bad_unit in ('deg', 'Degree', None):
        try:
            tools.getAngleBetween(x_axis, np.array([0., 1., 0.]),
                                  angle_unit=bad_unit)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Should reject angle_unit={bad_unit!r}')


def test_rotation_matrix():
    """The rotated z axis lies along the given angular momentum."""
    ang_mom = np.array([1., 2., 3.])
    mat = tools.getRotationTransformationMatrix(ang_mom)
    # A rotation, not a reflection: the determinant must be +1.
    assert np.isclose(np.linalg.det(mat), 1.)
    z_axis = mat[:, 2]
    assert np.allclose(z_axis, ang_mom / np.linalg.norm(ang_mom))
    # The new axes are orthonormal.
    assert np.allclose(np.dot(mat.T, mat), np.identity(3))
    # Coordinates round-trip through the new frame.
    points = np.array([[1., 2., 3.], [-4., 0.5, 2.]])
    assert np.allclose(np.dot(np.dot(points, mat), mat.T), points)
    # A vector perpendicular to the angular momentum keeps z = 0.
    mat = tools.getRotationTransformationMatrix(np.array([0., 0., 2.]))
    assert np.isclose(np.dot(np.array([1., 0., 0.]), mat)[2], 0.)
    # An angular momentum given as a Quantity is accepted.
    quantity = np.array([0., 0., 5.]) * units.kpc * units.km / units.s
    mat = tools.getRotationTransformationMatrix(quantity)
    assert np.allclose(np.dot(np.array([0., 0., 1.]), mat), [0., 0., 1.])
    # A negative angular momentum puts the new z axis the other way round.
    mat = tools.getRotationTransformationMatrix(-2.5 * ang_mom)
    unit = ang_mom / np.linalg.norm(ang_mom)
    assert np.allclose(mat[:, 2], -unit)
    assert np.isclose(np.linalg.det(mat), 1.)


def test_rotation_matrix_with_zero_ang_mom():
    """A zero angular momentum has no direction: warn, and return NaN."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mat = tools.getRotationTransformationMatrix(np.zeros(3))
    assert len(caught) == 1
    assert 'no direction' in str(caught[0].message)
    assert mat.shape == (3, 3)
    assert np.all(np.isnan(mat))


def test_rotation_matrix_when_ang_mom_is_the_preferred_axis():
    """An angular momentum along the preferred x axis still works."""
    mat = tools.getRotationTransformationMatrix(np.array([1., 0., 0.]))
    assert np.all(np.isfinite(mat))
    assert np.isclose(np.linalg.det(mat), 1.)
    assert np.allclose(mat[:, 2], [1., 0., 0.])


def test_plane_fitting():
    """A plane is recovered from points that lie exactly on it."""
    xs = np.array([0., 1., 2., 3.])
    ys = np.array([0., 1., 0., 1.])
    zs = 2. * xs - 3. * ys + 0.5
    param, normal = tools.leastSquarePlaneFitting(xs, ys, zs)
    assert np.allclose(param, [2., -3., 0.5])
    assert np.allclose(normal, [2., -3., -1.])
    # Three known planes, and a normal that is perpendicular to each one.
    for a, b, d in [(2., -3., 0.5), (0., 1., -2.), (1.5, 0., 3.)]:
        zs = a * xs + b * ys + d
        param, normal = tools.leastSquarePlaneFitting(xs, ys, zs)
        assert np.allclose(param, [a, b, d])
        # Two vectors that span the plane are (1, 0, a) and (0, 1, b).
        assert np.isclose(np.dot(normal, [1., 0., a]), 0.)
        assert np.isclose(np.dot(normal, [0., 1., b]), 0.)


def test_plane_fitting_on_noise():
    """A noisy plane is recovered close to the truth."""
    rng = np.random.default_rng(3)
    xs = rng.uniform(0, 10, 200)
    ys = rng.uniform(0, 10, 200)
    zs = 0.7 * xs - 1.2 * ys + 4. + rng.normal(scale=0.01, size=200)
    param, _ = tools.leastSquarePlaneFitting(xs, ys, zs)
    assert np.allclose(param, [0.7, -1.2, 4.], atol=0.05)
    # The same answer as numpy's own least-squares solve.
    design = np.c_[xs, ys, np.ones(xs.shape[0])]
    reference, *_ = np.linalg.lstsq(design, zs, rcond=None)
    assert np.allclose(param, reference)


def test_contain_value_radius():
    """Return the radius that holds the target value of the particles."""
    # Ten particles of equal value at radius 1, 2, ..., 10.
    re_dist = np.arange(1., 11.)
    value = np.ones(10)
    # The answer is the distance of the particle that reaches the target,
    # not of the next one.
    radius = tools.getContainCertainValueRadius(re_dist, value, 5.)
    assert radius == 5.
    # A target between two particles takes the farther one.
    radius = tools.getContainCertainValueRadius(re_dist, value, 5.5)
    assert radius == 6.
    too_big = tools.getContainCertainValueRadius(re_dist, value, 99.)
    assert np.isnan(too_big)
    # A target equal to the total takes the last particle.
    assert tools.getContainCertainValueRadius(re_dist, value, 10.) == 10.
    # The order of the particles does not matter.
    shuffled = np.array([7., 2., 9., 1., 5., 3., 8., 10., 4., 6.])
    assert tools.getContainCertainValueRadius(shuffled, value, 5.) == 5.
    # The inner particle wins when it holds the target alone.
    radius = tools.getContainCertainValueRadius(
        np.array([1., 2.]), np.array([9., 1.]), 5.)
    assert radius == 1.
    # Duplicated distances do not break the search.
    radius = tools.getContainCertainValueRadius(
        np.array([1., 1., 1., 2.]), np.ones(4), 2.)
    assert radius == 1.
    # The unit of a Quantity input is kept.
    radius = tools.getContainCertainValueRadius(
        np.array([1., 2., 3.]) * units.kpc, np.ones(3), 2.)
    assert radius.unit == units.kpc and radius.value == 2.


def test_radius_of_uniform_sphere():
    """Half the mass of a uniform sphere lies within 0.5^(1/3) of R."""
    rng = np.random.default_rng(4)
    radius_max = 10.
    count = 200000
    # Uniform in the volume: r = R * u^(1/3) with u uniform in [0, 1].
    re_dist = radius_max * rng.uniform(0, 1, count) ** (1. / 3.)
    mass = np.ones(count)
    radius = tools.getContainCertainValueRadius(re_dist, mass, 0.5 * count)
    analytic = 0.5 ** (1. / 3.) * radius_max
    assert abs(radius - analytic) < 0.02 * analytic


def test_contain_value_radius_rejects_bad_input():
    """No particles, or a value that is not finite, give NaN."""
    # An empty particle set has no radius, and does not crash.
    assert np.isnan(tools.getContainCertainValueRadius(
        np.array([]), np.array([]), 0.5))
    # A NaN in the values makes the radius undefined.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        radius = tools.getContainCertainValueRadius(
            np.array([1., 2., 3.]), np.array([1., np.nan, 1.]), 1.5)
    assert np.isnan(radius)
    assert len(caught) == 1
    assert 'not finite' in str(caught[0].message)
    # The unit is kept, even when the answer is NaN.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        radius = tools.getContainCertainValueRadius(
            np.array([1., 2.]) * units.kpc, np.array([np.inf, 1.]), 0.5)
    assert np.isnan(radius.value) and radius.unit == units.kpc


def test_contain_fraction_radius():
    """The fraction of the value is contained, and the unit is kept."""
    re_dist = np.arange(1., 11.) * units.kpc
    value = np.ones(10)
    radius = tools.getContainCertainFractionRadius(re_dist, value, 0.5)
    assert radius.unit == units.kpc
    assert radius.value == 5.


def test_contain_fraction_radius_grows_with_the_fraction():
    """More of the total means a larger radius, and at least the target."""
    rng = np.random.default_rng(5)
    re_dist = rng.uniform(0, 10, 5000)
    mass = rng.uniform(0.5, 2., 5000)
    radii = np.array([tools.getContainCertainFractionRadius(
        re_dist, mass, fraction) for fraction in (0.1, 0.5, 0.9)])
    assert np.all(np.diff(radii) > 0)
    # The fraction version agrees with the value version.
    total = mass.sum()
    assert np.isclose(
        tools.getContainCertainFractionRadius(re_dist, mass, 0.5),
        tools.getContainCertainValueRadius(re_dist, mass, 0.5 * total))
    # The returned radius encloses at least the target fraction.
    order = np.argsort(re_dist)
    cumulative = np.cumsum(mass[order])
    index = int(np.searchsorted(cumulative, 0.5 * total, side='left'))
    assert cumulative[index] >= 0.5 * total


def test_absolute_magnitude_to_luminosity():
    """A fainter magnitude gives a lower luminosity."""
    sun = tools.absoluteMagnitudeToLuminosity(np.array(4.83))
    assert np.isclose(sun, const.L_sun.to(units.W).value)
    bright = tools.absoluteMagnitudeToLuminosity(np.array(0.))
    faint = tools.absoluteMagnitudeToLuminosity(np.array(10.))
    assert bright > faint
    # Five magnitudes are a factor of 100 in luminosity.
    zero = tools.absoluteMagnitudeToLuminosity(np.array(0.))
    five = tools.absoluteMagnitudeToLuminosity(np.array(5.))
    assert np.isclose(zero / five, 100.)
    # The conversion round-trips with the magnitude formula.
    magnitudes = np.array([-2., 0., 4.83, 10., 20.])
    luminosities = tools.absoluteMagnitudeToLuminosity(magnitudes)
    back = 4.83 - 2.5 * np.log10(luminosities / const.L_sun.to(units.W).value)
    assert np.allclose(back, magnitudes)
    assert np.all(luminosities > 0.)


def test_pbc_relative_coord():
    """Relative coordinates are wrapped into half the box length."""
    box_length = 10.
    relative_coord = np.array([[6., -6., 0.],
                               [-4., 4., 5.],
                               [14., -14., 3.]])
    original = relative_coord.copy()
    out = tools.processingPbcRelativeCoord(relative_coord, box_length)
    assert np.allclose(out, [[-4., 4., 0.],
                             [-4., 4., 5.],
                             [4., -4., 3.]])
    # The array that was passed in is left alone.
    assert np.array_equal(relative_coord, original)
    # Analytic wrapping to the nearest image, at any distance.
    for given, want in [(0., 0.), (4., 4.), (-4., -4.), (6., -4.),
                        (-6., 4.), (14., 4.), (-14., -4.), (5., 5.),
                        (-5., -5.), (16., -4.), (-16., 4.), (24., 4.),
                        (-24., -4.), (105., 5.)]:
        given_coord = np.array([[given, 0., 0.]])
        got = tools.processingPbcRelativeCoord(given_coord, 10.)
        assert np.isclose(got[0, 0], want), (given, got[0, 0])
    # The nearest image is never farther than half a box length.
    grid = np.arange(-40., 40.01, 0.5)
    wrapped = tools.processingPbcRelativeCoord(np.c_[grid, grid, grid], 10.)
    assert np.all(np.abs(wrapped) <= 5. + 1e-12)
    # A Quantity stays a Quantity, and an integer array becomes a float one.
    quantity = np.array([[16., -6., 0.]]) * units.kpc
    assert tools.processingPbcRelativeCoord(quantity, 10.).unit == units.kpc
    integer_coord = np.array([[16, -6, 0]])
    assert tools.processingPbcRelativeCoord(integer_coord, 10.).dtype == float
    # Wrapping a point that is already inside changes nothing.
    once = tools.processingPbcRelativeCoord(np.array([[1., -2., 3.]]), 10.)
    twice = tools.processingPbcRelativeCoord(once, 10.)
    assert np.allclose(once, twice)


def test_pbc_coor():
    """Coordinates end up within half a box length of the center."""
    box_length = 10.
    center = np.array([9.5, 9.5, 9.5])
    coordinates = np.array([[0.5, 0.5, 0.5],  # wraps across the box edge
                            [9.0, 9.0, 9.0]])
    original = coordinates.copy()
    out = tools.processingPbcCoord(coordinates, center, box_length)
    assert np.array_equal(coordinates, original)
    offset = out - center
    assert np.all(np.abs(offset) <= box_length / 2.)
    assert np.allclose(offset[0], [1., 1., 1.])
    # A point just across the box edge is close to the center.
    center = np.array([0.2, 0.2, 0.2])
    point = np.array([[9.9, 0.2, 0.2]])
    out = tools.processingPbcCoord(point, center, box_length)
    assert np.isclose(np.linalg.norm(out - center), 0.3)
    # Every point stays within half a box length of the center.
    rng = np.random.default_rng(6)
    coordinates = rng.uniform(0, box_length, (1000, 3))
    for center in ([5., 5., 5.], [0.5, 0.5, 0.5], [9.5, 9.5, 9.5]):
        out = tools.processingPbcCoord(
            coordinates, np.array(center), box_length)
        assert np.all(np.abs(out - np.array(center)) <= box_length / 2. + 1e-9)


def test_pbc_distance_matches_the_torus():
    """The wrapped distance is the true distance on the periodic box."""
    box_length = 75000.
    rng = np.random.default_rng(11)
    for center in ([0., 0., 0.], [74999., 74999., 74999.],
                   [1., 37499., 37500.]):
        center = np.array(center)
        coordinates = rng.uniform(0, box_length, (500, 3))
        wrapped = tools.processingPbcCoord(coordinates, center, box_length)
        got = np.linalg.norm(wrapped - center, axis=1)
        # Reference: the smallest distance over the 27 neighbouring images.
        best = np.full(coordinates.shape[0], np.inf)
        for shift in itertools.product((-1, 0, 1), repeat=3):
            moved = coordinates - center - np.array(shift) * box_length
            best = np.minimum(best, np.linalg.norm(moved, axis=1))
        assert np.allclose(got, best)


def test_pbc_radius_is_frame_independent():
    """R50 of one system does not depend on where the box edge falls."""
    box_length = 75000.
    rng = np.random.default_rng(13)
    offset = rng.uniform(-30, 30, (2000, 3))
    offset = offset[np.linalg.norm(offset, axis=1) < 30.]
    mass = rng.uniform(0.5, 2., offset.shape[0])
    radii = []
    for center in ([100., 100., 100.], [37400., 200., 74900.]):
        center = np.array(center)
        coordinates = (center + offset) % box_length
        wrapped = tools.processingPbcCoord(coordinates, center, box_length)
        distance = np.linalg.norm(wrapped - center, axis=1)
        radii.append(tools.getContainCertainFractionRadius(
            distance, mass, 0.5))
    assert np.isclose(radii[0], radii[1], rtol=1e-12)


def test_parallel_function():
    """Inputs are passed as tuples or as single values, in order."""
    # An input tuple is unpacked into the positional arguments.
    outputs = tools.parallelFunction(
        pow, [(2, 3), (3, 2)], parallel=False, verbose=20)
    assert outputs == [8, 9]
    # A single value is passed as the only argument.
    outputs = tools.parallelFunction(
        len, ['a', 'bb', 'ccc'], parallel=False, verbose=20)
    assert outputs == [1, 2, 3]
    # The parallel path returns the same outputs in the same order.
    outputs = tools.parallelFunction(
        pow, [(2, 3), (3, 2)], parallel=True, n_jobs=2, verbose=20)
    assert outputs == [8, 9]
    # Parallel and serial agree, and both keep the order of the inputs.
    inputs = [(idx, idx + 1) for idx in range(20)]
    serial = tools.parallelFunction(pow, inputs, parallel=False, verbose=20)
    parallel = tools.parallelFunction(pow, inputs, parallel=True, n_jobs=2,
                                      verbose=20)
    assert serial == parallel == [i ** (i + 1) for i in range(20)]
    # An empty input list gives an empty output list.
    assert tools.parallelFunction(pow, [], parallel=False, verbose=20) == []
    # A TypeError from inside the function is not swallowed and retried.

    def raises_inside(x):
        return x + None

    try:
        tools.parallelFunction(
            raises_inside, [(1,)], parallel=False, verbose=20)
    except TypeError as error:
        assert 'NoneType' in str(error)
    else:
        raise AssertionError('The TypeError should reach the caller')


if __name__ == '__main__':
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith('test_') and callable(obj)]
    for check in checks:
        check()
        print(f'PASS {check.__name__}')
    print(f'{len(checks)} checks passed')
