"""
Read data from the Supplementary Data Catalogs.
For more details, see
https://www.tng-project.org/data/docs/specifications/#sec5

The supplementary catalogs should be downloaded and stored under
``~/data/simulations/<simName>/supplementary/``.

The functions apply to the following catalogs:

(c) stellar_circs      circularities, angular momenta, axis ratios of stars
(g) stellar_assembly   in situ and ex situ stellar mass growth
(q) halo_structure     concentration, shape, formation time of FoF halos

Every function raises FileNotFoundError when a catalog's file is missing,
and KeyError when the requested subhalo or halo is not listed.
"""

import warnings
from contextlib import contextmanager
from pathlib import Path
import h5py
import numpy as np
from ..accessdata import utils

_SPEC_URL = 'https://www.tng-project.org/data/docs/specifications/'

# Fields of an entry, keyed by the name of the catalog:
#   directory       directory under supplementary/
#   filename        template of the file name, filled with simName and snapnum
#   group           group that holds one snapshot, None when the file is flat
#   index           dataset that holds the object ID of a row, None when the
#                   row number is the object ID
#   section         section of the specifications
#   note            line added to the error when the file is missing
#
# simName is the simulation name in lower case. A template that carries the
# snapshot number writes it as {snapnum:03d}, because the release pads the 100
# snapshots of a run to three digits; a template that does not carry it leaves
# it out, because one file then holds every snapshot.
_CATALOGS = {
    'stellar_circs': {
        'directory': 'stellar_circularities',
        'filename': '{simName}_stellar_circs.hdf5',
        'group': 'Snapshot_{snapnum:d}',
        'index': 'SubfindID',
        'section': 'sec5c',
        'note': '',
    },
    'stellar_assembly': {
        'directory': 'stellar_assembly',
        'filename': '{simName}_stellar_assembly.hdf5',
        'group': 'Snapshot_{snapnum:d}',
        'index': None,
        'section': 'sec5g',
        'note': '',
    },
    'halo_structure': {
        'directory': 'halo_structure',
        'filename': '{simName}_halo_structure_{snapnum:03d}.hdf5',
        'group': None,
        'index': None,
        'section': 'sec5q',
        'note': 'Only the highest-resolution run of each box (TNG50-1, '
                'TNG100-1, TNG300-1 and their dark-matter-only counterparts) '
                'has this catalog, and only for the 20 full snapshots.',
    },
}


def getSubhaloesStellarCirc(simName, snapnum):
    """
    Read the stellar circularities of all subhaloes in one snapshot.

    Notes
    -----
    Catalog (c). Only subhaloes above a stellar mass threshold are listed, so
    a row number is not a subhalo number: look the subhalo up in the
    ``SubfindID`` array, which is returned with the rest.

    See https://www.tng-project.org/data/docs/specifications/#sec5c
    """
    return _readCatalog(simName, snapnum, 'stellar_circs')


def getSubhaloStellarCirc(simName, snapnum, sfid):
    """
    Read the stellar circularities of one subhalo in one snapshot.

    Notes
    -----
    Catalog (c), see :func:`getSubhaloesStellarCirc`. ``sfid`` is a Subfind
    ID, not a row number, and a subhalo that the catalog does not list raises
    KeyError.

    See https://www.tng-project.org/data/docs/specifications/#sec5c
    """
    return _readRow(simName, snapnum, 'stellar_circs', sfid)


def getSubhaloesStellarAssembly(simName, snapnum):
    """
    Read the stellar assembly of all subhaloes in one snapshot.

    Notes
    -----
    Catalog (g). One row per subhalo, so a row number is a subhalo number.
    This catalog is large: about 260 MB per snapshot and 26 GB for the whole
    file of TNG100-1.

    See https://www.tng-project.org/data/docs/specifications/#sec5g
    """
    return _readCatalog(simName, snapnum, 'stellar_assembly')


def getSubhaloStellarAssembly(simName, snapnum, sfid):
    """
    Read the stellar assembly of one subhalo in one snapshot.

    Notes
    -----
    Catalog (g), see :func:`getSubhaloesStellarAssembly`. Only the requested
    row is read. A subhalo outside the catalog raises KeyError.

    See https://www.tng-project.org/data/docs/specifications/#sec5g
    """
    return _readRow(simName, snapnum, 'stellar_assembly', sfid)


def getHaloesStructure(simName, snapnum):
    """
    Read the structural properties of all FoF haloes in one snapshot.

    Notes
    -----
    Catalog (q). A row number is a FoF group number.

    See https://www.tng-project.org/data/docs/specifications/#sec5q
    """
    return _readCatalog(simName, snapnum, 'halo_structure')


def getHaloStructure(simName, snapnum, fofid):
    """
    Read the structural properties of one FoF halo in one snapshot.

    Notes
    -----
    Catalog (q), see :func:`getHaloesStructure`. Only the requested row is
    read. A halo outside the catalog raises KeyError; a halo with
    ``GroupFlag == 0`` issues a warning and returns NaN.

    See https://www.tng-project.org/data/docs/specifications/#sec5q
    """
    halo = _readRow(simName, snapnum, 'halo_structure', fofid)
    if halo['GroupFlag'] == 0:
        warnings.warn(f'{simName} snapshot {snapnum}: halo {fofid} has '
                      'GroupFlag == 0, it is below the mass threshold of the '
                      'catalog and its properties were not computed',
                      stacklevel=2)
    return halo


def _getCatalogPath(simName, snapnum, catalog):
    """
    Build the path of the file that holds one catalog for one snapshot.

    The catalogs sit next to ``output/`` under the simulation directory, so
    the root comes from ``utils.getSimBasePath`` and only the levels below it
    are added here.
    """
    spec = _CATALOGS[catalog]
    root = Path(utils.getSimBasePath(simName)).parent
    name = spec['filename'].format(simName=simName.lower(), snapnum=snapnum)
    return root / 'supplementary' / spec['directory'] / name


@contextmanager
def _openDatasets(simName, snapnum, catalog):
    """ Open one catalog of one snapshot and yield its datasets by name. """
    spec = _CATALOGS[catalog]
    where = f'{simName} snapshot {snapnum}'
    path = _getCatalogPath(simName, snapnum, catalog)
    if not path.exists():
        lines = [f'{where}: no {catalog} in the local data.',
                 f'Expected file: {path}']
        if spec['note']:
            lines.append(spec['note'])
        lines.append(f'See {_SPEC_URL}#{spec["section"]}')
        raise FileNotFoundError('\n'.join(lines))
    with h5py.File(path, 'r') as h5file:
        node = h5file
        if spec['group'] is not None:
            name = spec['group'].format(snapnum=snapnum)
            if name not in h5file:
                raise KeyError(f'{where}: no group {name!r} in the {catalog} '
                               'catalog')
            node = h5file[name]
            if not isinstance(node, h5py.Group):
                raise KeyError(f'{where}: {name!r} is not a group in the '
                               f'{catalog} catalog')
        yield {name: obj for name, obj in node.items()
               if isinstance(obj, h5py.Dataset)}


def _locateRow(datasets, spec, index, where):
    """
    Find the row of a node that holds one object.

    Catalog (c) lists only subhaloes above a mass threshold, so its rows are
    found through the ``SubfindID`` dataset. Catalogs (g) and (q) list every
    subhalo or halo in order, so the row is the object number itself and only
    a range check is needed.
    """
    if spec['index'] is None:
        size = next(iter(datasets.values())).shape[0]
        if not 0 <= index < size:
            raise KeyError(f'{where}: {index} is outside the catalog, which '
                           f'lists {size} objects')
        return index
    ids = datasets[spec['index']][...]
    found = np.flatnonzero(ids == index)
    if found.size == 0:
        raise KeyError(f'{where}: {index} is not in the catalog, which lists '
                       f'{len(ids)} objects')
    return int(found[0])


def _readCatalog(simName, snapnum, catalog):
    """ Read every dataset of one catalog for one snapshot. """
    with _openDatasets(simName, snapnum, catalog) as datasets:
        return {name: dataset[...] for name, dataset in datasets.items()}


def _readRow(simName, snapnum, catalog, index):
    """ Read one row of a catalog, without reading the whole snapshot. """
    where = f'{simName} snapshot {snapnum}: {catalog}'
    with _openDatasets(simName, snapnum, catalog) as datasets:
        row = _locateRow(datasets, _CATALOGS[catalog], index, where)
        return {name: dataset[row] for name, dataset in datasets.items()}
