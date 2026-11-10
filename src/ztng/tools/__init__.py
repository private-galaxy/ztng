"""Small helpers that several modules share.

    linalg          vector and matrix helpers
    statistics      radii that contain a value or a fraction of it
    coords          positions under the periodic boundary conditions
    parallel        run one function over many inputs
    astrophysics    magnitude and luminosity
"""

from .astrophysics import absoluteMagnitudeToLuminosity
from .coords import processingPbcCoord, processingPbcRelativeCoord
from .linalg import (
    divide,
    getAngleBetween,
    getRotationTransformationMatrix,
    leastSquarePlaneFitting,
    log10,
)
from .parallel import parallelFunction
from .statistics import (
    getContainCertainFractionRadius,
    getContainCertainValueRadius,
)

__all__ = [
    'absoluteMagnitudeToLuminosity',
    'divide',
    'getAngleBetween',
    'getContainCertainFractionRadius',
    'getContainCertainValueRadius',
    'getRotationTransformationMatrix',
    'leastSquarePlaneFitting',
    'log10',
    'parallelFunction',
    'processingPbcCoord',
    'processingPbcRelativeCoord',
]
