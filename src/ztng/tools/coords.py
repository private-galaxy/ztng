"""Positions under the periodic boundary conditions (PBC).

See https://en.wikipedia.org/wiki/Periodic_boundary_conditions
"""

import numpy as np


def processingPbcRelativeCoord(relative_coord, box_length):
    """Wrap relative coordinates to the nearest image in a new array.

    Uses the minimum image convention: x - box_length * round(x / box_length).
    A box_length that is not finite means no wrapping.
    """
    relative_coord = np.asanyarray(relative_coord)
    if not np.isfinite(box_length):
        return relative_coord.copy()
    return relative_coord - box_length * np.round(relative_coord / box_length)


def processingPbcRelativeDist(coordinates, center_position, box_length):
    """Distance from the center, with the minimum image convention."""
    re_coor = processingPbcRelativeCoord(coordinates - center_position,
                                         box_length)
    return np.linalg.norm(re_coor, axis=1)


def processingPbcCoord(coordinates, center_position, box_length):
    """Wrap coordinates around a center and return a new array."""
    relative_coord = coordinates - center_position
    relative_coord = processingPbcRelativeCoord(relative_coord, box_length)
    coordinates = relative_coord + center_position
    return coordinates
