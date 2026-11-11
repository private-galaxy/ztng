"""
Read the IllustrisTNG data from the local files of the data release.

Every function is a thin wrapper around ``illustris_python``, which expects
the directory layout described in the docstring of ``utils``.

See https://www.tng-project.org/data/docs/scripts/ for the example scripts.
"""

import illustris_python as il
from .utils import getSimBasePath


def _checkPartType(partType):
    """ Raise ValueError for a particle type that illustris_python rejects. """
    try:
        il.util.partTypeNum(partType)
    except Exception as err:
        raise ValueError(f'unknown partType {partType!r}') from err


def getSnapHeader(simName, snapnum):
    """ Get the group catalog header for a snapshot. """
    basePath = getSimBasePath(simName)
    return il.groupcat.loadHeader(basePath, snapnum)


def getSubhaloInfo(simName, snapnum, sfid):
    """ Get the group catalog information of this subhalo. """
    basePath = getSimBasePath(simName)
    return il.groupcat.loadSingle(basePath, snapnum, haloID=-1, subhaloID=sfid)


def getSubhaloSnap(simName, snapnum, sfid, partType, fields):
    """ Get the snapshot data of a specific type of particles. """
    _checkPartType(partType)
    basePath = getSimBasePath(simName)
    return il.snapshot.loadSubhalo(basePath, snapnum, sfid, partType, fields)


_ALLOWED_TREE_TYPES = {
    'sublink': {},
    'sublink_mpb': {'onlyMPB': True},
    'sublink_mdb': {'onlyMDB': True},
}


def getSubhaloTree(simName, snapnum, sfid, treeType):
    """ Get the merger tree of this subhalo. """
    if treeType not in _ALLOWED_TREE_TYPES:
        raise ValueError(
            f'treeType {treeType!r} is not available from local data; '
            f'allowed: {sorted(_ALLOWED_TREE_TYPES)}')
    basePath = getSimBasePath(simName)
    return il.sublink.loadTree(
        basePath, snapnum, sfid, **_ALLOWED_TREE_TYPES[treeType])


def getHaloInfo(simName, snapnum, fofid):
    """ Get the group catalog information of this halo. """
    basePath = getSimBasePath(simName)
    return il.groupcat.loadSingle(
        basePath, snapnum, haloID=fofid, subhaloID=-1)


def getHaloSnap(simName, snapnum, fofid, partType, fields):
    """ Get the snapshot data of a specific type of particles. """
    _checkPartType(partType)
    basePath = getSimBasePath(simName)
    return il.snapshot.loadHalo(basePath, snapnum, fofid, partType, fields)
