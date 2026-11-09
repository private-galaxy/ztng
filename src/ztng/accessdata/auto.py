"""
Read the IllustrisTNG data, preferring the local simulation files.

Every function first tries the local files of the IllustrisTNG data release
(see https://www.tng-project.org/data/docs/scripts/) and falls back to the
web-based API (see https://www.tng-project.org/data/docs/api/) when the local
copy is missing.
"""

from . import local
from . import online


def getSnapHeader(simName, snapnum):
    """ Get the group catalog header. """
    try:
        header = local.getSnapHeader(simName, snapnum)
    except FileNotFoundError:
        header = online.getSnapHeader(simName, snapnum)
    return header


def getSnapRedshift(simName, snapnum):
    """ Get the redshift corresponding to the given snapshot. """
    header = getSnapHeader(simName, snapnum)
    redshift = header['Redshift']
    return redshift


def getSnapLookbackTime(simName, snapnum):
    """ Get the lookback time [Gyr] corresponding to the given snapshot. """
    cosmo = online.getSimCosmology(simName)
    redshift = getSnapRedshift(simName, snapnum)
    lbtime = cosmo.lookback_time(redshift).value
    return lbtime


def getSubhaloInfo(simName, snapnum, sfid):
    """ Get the group catalog information of this subhalo. """
    try:
        info = local.getSubhaloInfo(simName, snapnum, sfid)
    except FileNotFoundError:
        info = online.getSubhaloInfo(simName, snapnum, sfid)
    return info


def getSubhaloSnap(simName, snapnum, sfid, partType, fields):
    """ Get the snapshot data of a specific type of particles. """
    try:
        return local.getSubhaloSnap(simName, snapnum, sfid, partType, fields)
    except FileNotFoundError:
        return online.getSubhaloSnap(simName, snapnum, sfid, partType, fields)


def getSubhaloTree(simName, snapnum, sfid, treeType):
    """ Get the merger tree of this subhalo. """
    try:
        return local.getSubhaloTree(simName, snapnum, sfid, treeType)
    except (FileNotFoundError, ValueError):
        return online.getSubhaloTree(simName, snapnum, sfid, treeType)


def getHaloInfo(simName, snapnum, fofid):
    """ Get the group catalog information of this halo. """
    try:
        info = local.getHaloInfo(simName, snapnum, fofid)
    except FileNotFoundError:
        info = online.getHaloInfo(simName, snapnum, fofid)
    return info


def getHaloSnap(simName, snapnum, fofid, partType, fields):
    """ Get the snapshot data of a specific type of particles. """
    try:
        return local.getHaloSnap(simName, snapnum, fofid, partType, fields)
    except FileNotFoundError:
        return online.getHaloSnap(simName, snapnum, fofid, partType, fields)
