"""
Access the IllustrisTNG data through the web-based API.

Each query first looks for a cached copy under the data directory returned by
``getSavedDataPath()``; see the docstring of ``utils`` for its default
location.  Saved metadata, group catalog information and merger trees are
returned without any request.  Saved snapshot data is reused only for the
fields it already holds: the missing fields are requested from the API and
merged into the same file.

See https://www.tng-project.org/data/docs/api/ for the API itself.
"""

import os
import h5py
import uuid
import shutil
import warnings
from typing import Any
import requests
import numpy as np
import illustris_python as il
from .utils import get
from .utils import allowed_partType
from .utils import partStandardName
from .utils import getSimUrl
from .utils import getSnapUrl
from .utils import getSubhaloUrl
from .utils import getSavedFileName
from .utils import getHaloUrl
from .utils import getSimBasePath


def _saveNpy(filename, obj):
    """ Save an object to a .npy file without leaving a partial file. """
    filename_tmp = filename.replace('.npy', f'_{uuid.uuid4()}.npy')
    try:
        np.save(filename_tmp, obj)
        os.replace(filename_tmp, filename)
    finally:
        if os.path.exists(filename_tmp):
            os.remove(filename_tmp)


def _saveBytes(filename, content):
    """ Write bytes to a file without leaving a partial file. """
    filename_tmp = f'{filename}_{uuid.uuid4()}.tmp'
    try:
        with open(filename_tmp, 'wb') as f:
            f.write(content)
        os.replace(filename_tmp, filename)
    finally:
        if os.path.exists(filename_tmp):
            os.remove(filename_tmp)


def getSimMeta(simName):
    """ Get the meta-data for the simulation. """
    simUrl = getSimUrl(simName)
    filename = getSavedFileName(file_type='simulation_metadata',
                                simName=simName)
    if not os.path.exists(filename):
        _saveNpy(filename, get(simUrl))
    return np.load(filename, allow_pickle=True).tolist()


# NOTE: astropy's module-level __getattr__ returns generic 'Cosmology', so
# pyright flags Planck15/FLRW's Om0/Ob0/h as errors; 'Any' is accurate until
# astropy declares these names properly.
def getSimCosmology(simName) -> Any:
    """ Get the cosmological calculations of this simulation. """
    sim_meta = getSimMeta(simName)
    if sim_meta['cosmology'] == 'Planck2015':
        from astropy.cosmology import Planck15 as cosmo
    else:
        raise ValueError(
            f'{simName}: the cosmology {sim_meta["cosmology"]!r} '
            f'is not supported')
    return cosmo


def getSimCosmicBaryonMassFraction(simName):
    """ Get the cosmic baryon mass fraction of this simulation. """
    cosmo = getSimCosmology(simName)
    cosmo_baryon_frac = cosmo.Ob0 / cosmo.Om0
    return cosmo_baryon_frac


def getSnapMeta(simName, snapnum):
    """ Get the meta-data for a snapshot. """
    filename = getSavedFileName(file_type='snapshot_metadata',
                                simName=simName,
                                snapnum=snapnum)
    if not os.path.exists(filename):
        snapUrl = getSnapUrl(simName, snapnum)
        _saveNpy(filename, get(snapUrl))
    return np.load(filename, allow_pickle=True).tolist()


def getSnapHeader(simName, snapnum):
    """ Get the group catalog header for a snapshot. """
    sim_meta = getSimMeta(simName)
    snap_meta = getSnapMeta(simName, snapnum)
    header = {}
    header['Time'] = 1. / (1. + snap_meta['redshift'])
    header['Redshift'] = snap_meta['redshift']
    header['HubbleParam'] = sim_meta['hubble']
    header['BoxSize'] = sim_meta['boxsize']
    header['Omega0'] = sim_meta['omega_0']
    header['OmegaLambda'] = sim_meta['omega_L']
    return header


def getSnapPhySoftening(simName, snapnum):
    """
    Get the softening length [kpc] of the snapshot.
    (Refer to Pillepichet al. 2018 for details:
    In TNG, the gravitational softenings for all particle types are
    comoving kpc (with value equal to that of the DM) down to z=1,
    after which they are fixed in physical space to their z=1 values.
    In Illustris, ...)
    """
    if 'TNG' not in simName:
        raise ValueError(
            f'{simName}: the calculation is only applicable to '
            f'TNG50/100/300.')
    sim_meta = getSimMeta(simName)
    header = getSnapHeader(simName, snapnum)
    # This is the softening of the collisionless component (dark matter,
    # stars and wind particles), which all share one value: comoving kpc/h
    # down to z=1, then fixed in physical space to the z=1 value
    # (Pillepich et al. 2018, Section 2.2.5).  Multiplying the comoving
    # value by the scale factor gives the physical one, and
    # 'softening_*_max_phys' is that fixed physical value; both are in units
    # of kpc/h, and dividing by h gives kpc.  Using min() reproduces the z=1
    # transition without assuming which snapshot number corresponds to z=1.
    #
    # Two particle types follow other rules:
    #   - gas cells are softened adaptively, eps_gas = 2.5 * r_cell with a
    #     floor eps_gas_min 8 times smaller than the comoving value here
    #     (the metadata holds 'softening_gas_factor' = 2.5 and
    #     'softening_gas_comoving_min' = 1/8 of 'softening_dm_comoving');
    #   - each black hole uses eps_DM * (m_BH / m_DM)**(1/3), which differs
    #     from one black hole to the next.
    #   https://www.tng-project.org/data/forum/topic/265/
    #   https://www.tng-project.org/data/forum/topic/408/
    softening = min(sim_meta['softening_dm_comoving'] * header['Time'],
                    sim_meta['softening_dm_max_phys'])  # kpc/h
    return softening / header['HubbleParam']  # kpc


_SNAPSHOT_PARTTYPES = ['gas', 'dm', 'tracer', 'stars', 'bhs']


def _getGroup(node, name):
    """ Get one group of an open HDF5 file. """
    grp = node[name]
    if not isinstance(grp, h5py.Group):
        raise TypeError(
            f'{name} of {node.file.filename} is not an HDF5 group; the cached '
            f'file is probably damaged, delete it and query again')
    return grp


def _getDataset(node, name):
    """ Get one dataset of an open HDF5 group. """
    dataset = node[name]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(
            f'{name} of {node.file.filename} is not an HDF5 dataset; the '
            f'cached file is probably damaged, delete it and query again')
    return dataset


def _loadSnap(file_type, **fileKeys):
    """ Load the fields that are already cached, creating the file if
        needed. """
    filename = getSavedFileName(file_type=file_type, **fileKeys)

    # create the snapshot data file of this object (if it doesn't exist)
    if not os.path.exists(filename):
        with h5py.File(filename, 'w') as f:
            for ptName in _SNAPSHOT_PARTTYPES:
                f.create_group(partStandardName(ptName))

    # load the snapshot data of this object
    with h5py.File(filename, 'r') as f:
        snap = {}
        for ptName in f:
            grp = _getGroup(f, ptName)
            snap[ptName] = {field: _getDataset(grp, field)[...]
                            for field in grp}
    return snap


def _getInfo(kind, simName, snapnum, objid):
    """ Get the group catalog information of this subhalo or halo. """
    if kind == 'subhalo':
        return getSubhaloInfo(simName, snapnum, objid)
    elif kind == 'halo':
        return getHaloInfo(simName, snapnum, objid)
    else:
        raise ValueError(
            f'kind {kind!r} is not supported; expected "subhalo" or "halo"')


def _getSnap(kind, simName, snapnum, objid, partType, fields):
    """ Get the snapshot data of a specific type of particles. """
    partType = allowed_partType(partType)
    ptStandardName = partStandardName(partType)
    if kind == 'subhalo':
        file_type = 'subhalo_snapshotdata'
        lenKey = 'SubhaloLenType'
        url = getSubhaloUrl(simName, snapnum, objid)
        # the keyword arguments that getSavedFileName builds the file name
        # from; the object identifier is called 'sfid' for a subhalo and
        # 'fofid' for a halo
        fileKeys = {'simName': simName, 'snapnum': snapnum, 'sfid': objid}
        haloID, subhaloID = -1, objid
    elif kind == 'halo':
        file_type = 'halo_snapshotdata'
        lenKey = 'GroupLenType'
        url = getHaloUrl(simName, snapnum, objid)
        fileKeys = {'simName': simName, 'snapnum': snapnum, 'fofid': objid}
        haloID, subhaloID = objid, -1
    else:
        raise ValueError(
            f'kind {kind!r} is not supported; expected "subhalo" or "halo"')

    filename = getSavedFileName(file_type=file_type, **fileKeys)
    filename_api = f'{filename}.api.{uuid.uuid4()}.hdf5'
    filename_new = f'{filename}.new.{uuid.uuid4()}.hdf5'

    # see which fields are already exist; an empty 'fields' means all of the
    # fields available for this particle type
    snap = _loadSnap(file_type, **fileKeys)[ptStandardName]
    downloadAll = not fields
    fields_requests = [field for field in fields if field not in snap]

    try:
        parts_len = None
        if fields_requests or downloadAll:
            try:
                info = il.groupcat.loadSingle(
                    getSimBasePath(simName), snapnum,
                    haloID=haloID, subhaloID=subhaloID)
            except FileNotFoundError:
                info = _getInfo(kind, simName, snapnum, objid)
            parts_len = info[lenKey][il.util.partTypeNum(partType)]

        if parts_len and (fields_requests or downloadAll):
            if downloadAll:
                warnings.warn(
                    f'{simName} snapshot {snapnum} {kind} {objid}: an empty '
                    f'field list means all {partType} fields, so the entire '
                    f'{partType} cutout will be downloaded', stacklevel=2)
                params = None
            else:
                params = {partType: ','.join(fields_requests)}
            try:
                r = get(os.path.join(url, 'cutout.hdf5'), params=params)
            except requests.HTTPError as err:
                raise ValueError(
                    f'{simName} snapshot {snapnum} {kind} {objid}: the TNG '
                    f'API rejected the {partType} cutout request for '
                    f'{fields_requests} ({err}); field names are case '
                    f'sensitive, and a field that does not exist for this '
                    f'particle type makes the whole request fail'
                ) from err

            # write the cutout data in to a temporary file, merge it into a
            # copy of the cached file, then replace the cached file
            # NOTE: the copy costs one full read and write of the cached
            # file; it buys a cache that is never left half written.
            with open(filename_api, 'wb') as f:
                f.write(r.content)
            with h5py.File(filename_api, 'r') as f_api:
                if ptStandardName in f_api:
                    apiGrp = _getGroup(f_api, ptStandardName)
                    shutil.copy2(filename, filename_new)
                    with h5py.File(filename_new, 'r+') as f_new:
                        grp = f_new.require_group(ptStandardName)
                        for field in apiGrp:
                            dataset = _getDataset(apiGrp, field)[...]
                            if field in grp:
                                _getDataset(grp, field)[...] = dataset
                            else:
                                grp[field] = dataset
                    os.replace(filename_new, filename)
    finally:
        # remove the temporary files
        for name in [filename_api, filename_new]:
            if os.path.exists(name):
                os.remove(name)

    # an object without particles of this type has no field to return, and
    # the fields of such an object are never cached
    if parts_len == 0:
        return {'count': 0}

    # reload the snapshot data
    snap = _loadSnap(file_type, **fileKeys)[ptStandardName]

    # return the result
    fields_return = list(snap.keys()) if downloadAll else fields
    missing = [field for field in fields_return if field not in snap]
    if missing:
        raise KeyError(
            f'{partType} fields {missing} are not available for '
            f'{simName} snapshot {snapnum} {kind} {objid}; '
            f'available: {sorted(snap)}')
    result = {field: snap[field] for field in fields_return}
    result['count'] = snap[fields_return[0]].shape[0] if fields_return else 0
    return result


def loadSubhaloSnap(simName, snapnum, sfid):
    """ Load all the fields that are already cached, for every particle type.

    Only the local cache is read; nothing is requested from the API.  An empty
    cache file is created when none exists yet. """
    return _loadSnap('subhalo_snapshotdata',
                     simName=simName, snapnum=snapnum, sfid=sfid)


def getSubhaloSnap(simName, snapnum, sfid, partType, fields):
    """ Get the snapshot data of a specific type of particles. """
    return _getSnap('subhalo', simName, snapnum, sfid, partType, fields)


def loadHaloSnap(simName, snapnum, fofid):
    """ Load all the fields that are already cached, for every particle type.

    Only the local cache is read; nothing is requested from the API.  An empty
    cache file is created when none exists yet. """
    return _loadSnap('halo_snapshotdata',
                     simName=simName, snapnum=snapnum, fofid=fofid)


def getHaloSnap(simName, snapnum, fofid, partType, fields):
    """ Get the snapshot data of a specific type of particles. """
    return _getSnap('halo', simName, snapnum, fofid, partType, fields)


def getSubhaloDetail(simName, snapnum, sfid):
    """ Get the subhalo detail. """
    filename = getSavedFileName(file_type='subhalo_detail',
                                simName=simName,
                                snapnum=snapnum,
                                sfid=sfid)
    if not os.path.exists(filename):
        subUrl = getSubhaloUrl(simName, snapnum, sfid)
        _saveNpy(filename, get(subUrl))
    return np.load(filename, allow_pickle=True).tolist()


def getSubhaloInfo(simName, snapnum, sfid):
    """ Get the group catalog information of this subhalo. """
    filename = getSavedFileName(file_type='subhalo_information',
                                simName=simName,
                                snapnum=snapnum,
                                sfid=sfid)
    if not os.path.exists(filename):
        subUrl = getSubhaloUrl(simName, snapnum, sfid)
        _saveNpy(filename, get(os.path.join(subUrl, 'info.json')))
    info = np.load(filename, allow_pickle=True).tolist()
    return info


def getSubhaloTree(simName, snapnum, sfid, treeType):
    """ Get the merger tree of this subhalo. """
    filename = getSavedFileName(file_type='subhalo_tree',
                                simName=simName,
                                snapnum=snapnum,
                                sfid=sfid,
                                treeType=treeType)

    if not os.path.exists(filename):
        detail = getSubhaloDetail(simName, snapnum, sfid)
        treeUrl = detail['trees'][treeType]
        if '.json' in treeUrl:
            _saveNpy(filename, get(treeUrl))
        if '.hdf5' in treeUrl:
            _saveBytes(filename, get(treeUrl).content)

    if treeType == 'sublink_simple':
        return np.load(filename, allow_pickle=True).tolist()
    else:
        tree = {}
        with h5py.File(filename, 'r') as f:
            for key in f:
                tree[key] = _getDataset(f, key)[...]
        return tree


def getHaloInfo(simName, snapnum, fofid):
    """ Get the group catalog information of this halo. """
    filename = getSavedFileName(file_type='halo_information',
                                simName=simName,
                                snapnum=snapnum,
                                fofid=fofid)
    if not os.path.exists(filename):
        haloUrl = getHaloUrl(simName, snapnum, fofid)
        _saveNpy(filename, get(os.path.join(haloUrl, 'info.json')))
    info = np.load(filename, allow_pickle=True).tolist()
    return info
