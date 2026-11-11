"""Access data for a group object (e.g., halo or subhalo)."""


from functools import cached_property
from ..accessdata import auto
from ..accessdata import online
from ..accessdata import utils


class HaloDataAccess:

    def __init__(self, simName, snapnum, fofid):
        self.simName = simName
        self.snapnum = snapnum
        self.fofid = fofid

    def getInfo(self):
        """ Get the group catalog information of this halo. """
        return auto.getHaloInfo(self.simName, self.snapnum, self.fofid)

    @cached_property
    def info(self):
        return self.getInfo()

    def getSnap(self, partType, fields):
        """ Get the snapshot data of a specific type of particles. """
        return auto.getHaloSnap(
            self.simName, self.snapnum, self.fofid, partType, fields)


class SubhaloDataAccess:

    def __init__(self, simName, snapnum, sfid):
        self.simName = simName
        self.snapnum = snapnum
        self.sfid = sfid

    def getURLs(self):
        """ Get the URLs of the subhalo. """
        return utils.getSubhaloUrls(self.simName, self.snapnum, self.sfid)

    @property
    def URLs(self):
        return self.getURLs()

    def getDetail(self):
        """ Get the subhalo detail. """
        return online.getSubhaloDetail(self.simName, self.snapnum, self.sfid)

    @property
    def detail(self):
        return self.getDetail()

    def getInfo(self):
        """ Get the group catalog information of this subhalo. """
        return auto.getSubhaloInfo(self.simName, self.snapnum, self.sfid)

    @cached_property
    def info(self):
        return self.getInfo()

    def getSnap(self, partType, fields):
        """ Get the snapshot data of a specific type of particles. """
        return auto.getSubhaloSnap(
            self.simName, self.snapnum, self.sfid, partType, fields)

    def getTree(self, treeType):
        """ Get the merger tree of this subhalo. """
        return auto.getSubhaloTree(
            self.simName, self.snapnum, self.sfid, treeType)
