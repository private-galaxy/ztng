"""Small helpers that several modules share.

    linalg          vector and matrix helpers
    statistics      radii that contain a value or a fraction of it
    coords          positions under the periodic boundary conditions
    parallel        run one function over many inputs
    astrophysics    magnitude, luminosity and cooling time
    arguments       check the keyword arguments of a function
    units           units of the TNG data
"""

from .arguments import updateKwargs
from .astrophysics import COOLING_TIME_CONVENTIONS
from .astrophysics import absoluteMagnitudeToLuminosity
from .astrophysics import getCoolingTime
from .astrophysics import internalEnergyToTemperature
from .coords import (
    processingPbcCoord,
    processingPbcRelativeCoord,
    processingPbcRelativeDist,
)
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
    getEnclosedMass,
)
from .units import (
    haloVelocityUnit,
    lengthUnit,
    massUnit,
    particleVelocityUnit,
    subhaloVelocityUnit,
)

__all__ = [
    'COOLING_TIME_CONVENTIONS',
    'absoluteMagnitudeToLuminosity',
    'divide',
    'getAngleBetween',
    'getContainCertainFractionRadius',
    'getContainCertainValueRadius',
    'getCoolingTime',
    'getEnclosedMass',
    'getRotationTransformationMatrix',
    'haloVelocityUnit',
    'internalEnergyToTemperature',
    'leastSquarePlaneFitting',
    'lengthUnit',
    'log10',
    'massUnit',
    'parallelFunction',
    'particleVelocityUnit',
    'processingPbcCoord',
    'processingPbcRelativeCoord',
    'processingPbcRelativeDist',
    'subhaloVelocityUnit',
    'updateKwargs',
]
