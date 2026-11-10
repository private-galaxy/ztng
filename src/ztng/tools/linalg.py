"""Vector and matrix helpers."""

import warnings

import numpy as np
from astropy import units


def log10(x1, default_out=np.nan):
    """Return log10 of x1, and default_out where x1 is not positive."""
    out = np.full_like(x1, default_out, dtype=float)
    return np.log10(x1, out=out, where=(x1 > 0.))


def divide(x1, x2, default_out=np.nan):
    """Return x1 / x2, and default_out where x2 is zero."""
    # Test for an exact zero on purpose: a tolerance would send a small
    # but non-zero denominator to default_out.
    nonzero = x2 != 0.  # noqa: RUF069
    out = np.full_like(x1, default_out, dtype=float)
    return np.divide(x1, x2, out=out, where=nonzero)


def getAngleBetween(vector_1, vector_2, angle_unit='rad'):
    """Return the angle between two vectors, in 'rad' or in 'degree'.
    Raise ValueError for any other unit.
    """
    vector_1 = divide(vector_1, np.linalg.norm(vector_1), default_out=np.nan)
    vector_2 = divide(vector_2, np.linalg.norm(vector_2), default_out=np.nan)
    cos_angle = np.dot(vector_1, vector_2)
    if isinstance(cos_angle, units.Quantity):
        # The division left a dimensionless Quantity, and arccos of such a
        # Quantity returns a Quantity in rad, so drop the unit here.
        cos_angle = cos_angle.value
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * units.rad
    if angle_unit == 'degree':
        angle = angle.to(unit=units.degree)
    elif angle_unit != 'rad':
        raise ValueError(
            f"angle_unit should be 'rad' or 'degree', not {angle_unit!r}.")
    return angle


def getRotationTransformationMatrix(
        ang_mom_vector, prefer_x_axis=(1., 0., 0.)):
    """Return the matrix that rotates to the frame of ang_mom_vector.
    The columns are the new axes and the determinant is +1, so
    np.dot(vector, matrix) is the vector in the new frame.
    """
    z_axis = ang_mom_vector
    z_axis = (z_axis * units.dimensionless_unscaled).value
    norm = np.linalg.norm(z_axis)
    # Test the divisor for an exact zero on purpose: a tolerance would
    # reject a small but well defined angular momentum.
    if norm == 0.:  # noqa: RUF069
        warnings.warn(
            'The angular momentum is zero, it has no direction, so the '
            'matrix is all NaN.')
        return np.full((3, 3), np.nan)
    z_axis = z_axis / norm
    y_axis = np.cross(prefer_x_axis, z_axis)
    if np.allclose(y_axis, 0.):
        # The preferred x axis is parallel to the angular momentum, so it
        # cannot fix the new x axis. Take the world axis that is least
        # aligned with the angular momentum instead.
        world_axes = np.identity(3)
        helper_axis = world_axes[np.argmin(np.abs(world_axes @ z_axis))]
        y_axis = np.cross(helper_axis, z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    trans_matrix = np.column_stack((x_axis, y_axis, z_axis))
    return trans_matrix


def leastSquarePlaneFitting(xs, ys, zs):
    """Fit z = a*x + b*y + d and return (a, b, d) with the normal.
    A vertical plane has no such form and cannot be fitted.
    """
    # Each row of the design matrix is [x_i, y_i, 1].
    mat = np.c_[xs, ys, np.ones(xs.shape[0])]
    param, _, _, _ = np.linalg.lstsq(mat, zs, rcond=None)
    # For the plane a*x + b*y - z + d = 0, the normal vector is (a, b, -1).
    normal = np.array([param[0], param[1], -1])
    return param, normal
