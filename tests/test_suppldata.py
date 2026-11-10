"""
Runnable checks of ztng.suppldata.

Usage:
    python tests/test_suppldata.py
    pytest tests/test_suppldata.py

Most checks build small HDF5 files that follow the specifications of the
catalogs, so they need neither the real data release nor a network. The last
check runs against the real files when they are on this machine, and returns
quietly otherwise.
"""

import os
import sys
import tempfile
import warnings

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ztng.accessdata import utils  # noqa: E402
from ztng.suppldata import local  # noqa: E402


class fakeData:
    """ Point the simulation base path at a temporary directory. """

    def __enter__(self):
        self.original = utils.getSimBasePath
        self.root = tempfile.mkdtemp()
        utils.getSimBasePath = lambda simName: os.path.join(
            self.root, simName, 'output/')
        return self

    def write(self, simName, catalog, snapnum, datasets, header=False):
        """
        Write one catalog file and return its path.

        Parameters
        ----------
        simName : str
            Name of the simulation.
        catalog : str
            Key of ``local._CATALOGS``.
        snapnum : int
            Snapshot number, used for the file and group names.
        datasets : dict
            One array per dataset to create.
        header : bool, optional
            Create the ``Header`` group that the real files carry.

        Returns
        -------
        pathlib.Path
            Path of the file that was written.
        """
        path = local._getCatalogPath(simName, snapnum, catalog)
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = local._CATALOGS[catalog]
        with h5py.File(path, 'w') as h5file:
            if header:
                h5file.create_group('Header').attrs['CatalogName'] = 'Test'
            if spec['group'] is None:
                node = h5file
            else:
                node = h5file.create_group(
                    spec['group'].format(snapnum=snapnum))
            for name, values in datasets.items():
                node.create_dataset(name, data=values)
        return path

    def __exit__(self, *exc):
        utils.getSimBasePath = self.original
        return False


CIRC = {
    'SubfindID': np.array([0, 1, 3, 7], dtype=np.int32),
    'CircAbove07Frac': np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    'SpecificAngMom': np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
}

HALO = {
    'GroupFlag': np.array([1, 1, 0, 0], dtype=np.int64),
    'M200c': np.array([14.0, 13.0, np.nan, np.nan]),
    'c200c': np.array([5.0, np.nan, np.nan, np.nan]),
    'Mean_vel': np.zeros((4, 3)),
}


def _field(rows, name):
    """ Build a named array whose values encode the row number. """
    return np.arange(rows, dtype=np.float64) + len(name)


def test_catalog_paths():
    """ The three catalogs resolve to the paths of the data release. """
    with fakeData():
        paths = {name: str(local._getCatalogPath('TNG100-1', 99, name))
                 for name in local._CATALOGS}
        circ, assembly, halo = (paths['stellar_circs'],
                                paths['stellar_assembly'],
                                paths['halo_structure'])
        expected = 'TNG100-1/supplementary/'
        for path, tail in ((circ, expected + 'stellar_circularities/'
                            'tng100-1_stellar_circs.hdf5'),
                           (assembly, expected +
                            'stellar_assembly/tng100-1_stellar_assembly.hdf5'),
                           (halo, expected + 'halo_structure/'
                            'tng100-1_halo_structure_099.hdf5')):
            if not path.endswith(tail):
                raise AssertionError(f'{path} does not end with {tail}')
        if '/output/' in circ:
            raise AssertionError('the catalog path must not keep output/')


def test_missing_file():
    """ A missing catalog raises FileNotFoundError with the expected path. """
    with fakeData():
        try:
            local.getHaloesStructure('TNG100-1', 99)
        except FileNotFoundError as error:
            text = str(error)
        else:
            raise AssertionError('should raise FileNotFoundError')
        for piece in ('TNG100-1 snapshot 99',
                      'halo_structure',
                      'halo_structure/tng100-1_halo_structure_099.hdf5',
                      'sec5q',
                      'full snapshots'):
            if piece not in text:
                raise AssertionError(f'{piece!r} is missing from {text!r}')


def test_bulk_read():
    """ Every dataset is read, and Header is not mistaken for a dataset. """
    with fakeData() as data:
        data.write('TNG100-1', 'stellar_circs', 57, CIRC, header=True)
        reader = local.getSubhaloesStellarCirc('TNG100-1', 57)
        if sorted(reader) != sorted(CIRC):
            raise AssertionError(f'got {sorted(reader)}')
        for name in CIRC:
            if not np.array_equal(reader[name], CIRC[name]):
                raise AssertionError(f'{name} does not round trip')


def test_flat_catalog():
    """ A catalog without snapshot groups is read from the file root. """
    with fakeData() as data:
        data.write('TNG100-1', 'halo_structure', 99, HALO, header=True)
        haloes = local.getHaloesStructure('TNG100-1', 99)
        if sorted(haloes) != sorted(HALO):
            raise AssertionError(f'got {sorted(haloes)}')
        if not np.array_equal(haloes['GroupFlag'], HALO['GroupFlag']):
            raise AssertionError('GroupFlag does not round trip')


def test_row_is_looked_up_by_id():
    """ A row is found through SubfindID, not by its position. """
    with fakeData() as data:
        data.write('TNG100-1', 'stellar_circs', 57, CIRC)
        row = local.getSubhaloStellarCirc('TNG100-1', 57, 7)
        if row['SpecificAngMom'] != 40.0:
            raise AssertionError(f'got {row["SpecificAngMom"]}')
        if row['SubfindID'] != 7:
            raise AssertionError(f'got {row["SubfindID"]}')


def test_row_matches_bulk():
    """ Reading one row gives the same values as reading the snapshot. """
    with fakeData() as data:
        data.write('TNG100-1', 'stellar_circs', 57, CIRC)
        bulk = local.getSubhaloesStellarCirc('TNG100-1', 57)
        row = local.getSubhaloStellarCirc('TNG100-1', 57, 3)
        position = int(np.flatnonzero(bulk['SubfindID'] == 3)[0])
        for name in bulk:
            if not np.array_equal(row[name], bulk[name][position]):
                raise AssertionError(f'{name} differs between the two paths')


def test_row_not_listed():
    """ A subhalo that the catalog does not list raises KeyError. """
    with fakeData() as data:
        data.write('TNG100-1', 'stellar_circs', 57, CIRC)
        try:
            local.getSubhaloStellarCirc('TNG100-1', 57, 2)
        except KeyError as error:
            text = str(error)
            if 'stellar_circs' not in text or 'not in the catalog' not in text:
                raise AssertionError(f'unexpected message: {error}')
        else:
            raise AssertionError('should raise KeyError')


def test_row_out_of_range():
    """ An index outside a catalog without SubfindID raises KeyError. """
    with fakeData() as data:
        data.write('TNG100-1', 'halo_structure', 99, HALO)
        for index in (4, -1):
            try:
                local.getHaloStructure('TNG100-1', 99, index)
            except KeyError as error:
                text = str(error)
                if 'halo_structure' not in text or 'outside' not in text:
                    raise AssertionError(f'unexpected message: {error}')
            else:
                raise AssertionError(f'{index} should raise KeyError')


def test_group_flag_warning():
    """ A halo below the mass threshold warns and still returns its row. """
    with fakeData() as data:
        data.write('TNG100-1', 'halo_structure', 99, HALO)
        halo = local.getHaloStructure('TNG100-1', 99, 0)
        if halo['GroupFlag'] != 1:
            raise AssertionError('halo 0 should be above the threshold')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            low = local.getHaloStructure('TNG100-1', 99, 3)
        if len(caught) != 1:
            raise AssertionError(f'expected one warning, got {len(caught)}')
        if 'GroupFlag == 0' not in str(caught[0].message):
            raise AssertionError(f'unexpected message: {caught[0].message}')
        if not np.isnan(low['M200c']):
            raise AssertionError('the row of halo 3 should hold NaN')


def test_missing_snapshot_group():
    """ A snapshot that the file does not hold raises KeyError. """
    with fakeData() as data:
        data.write('TNG100-1', 'stellar_circs', 57, CIRC)
        try:
            local.getSubhaloesStellarCirc('TNG100-1', 58)
        except KeyError as error:
            if 'no group' not in str(error):
                raise AssertionError(f'unexpected message: {error}')
        else:
            raise AssertionError('should raise KeyError')


REAL_CASES = [('TNG100-1', 'stellar_circs', 57),
              ('TNG100-1', 'halo_structure', 99),
              ('TNG50-4', 'stellar_assembly', 99)]


def test_real_catalogs():
    """ Read the real catalogs when they are on this machine. """
    checked = 0
    for simName, catalog, snapnum in REAL_CASES:
        path = local._getCatalogPath(simName, snapnum, catalog)
        if not path.exists():
            continue
        try:
            if catalog == 'stellar_circs':
                bulk = local.getSubhaloesStellarCirc(simName, snapnum)
                sfid = int(bulk['SubfindID'][0])
                row = local.getSubhaloStellarCirc(simName, snapnum, sfid)
                if row['SubfindID'] != sfid:
                    raise AssertionError('the row holds another subhalo')
            elif catalog == 'halo_structure':
                bulk = local.getHaloesStructure(simName, snapnum)
                if bulk['GroupFlag'][0] != 1:
                    raise AssertionError('halo 0 should be in the catalog')
                row = local.getHaloStructure(simName, snapnum, 0)
                if row['M200c'] != bulk['M200c'][0]:
                    raise AssertionError('the row differs from the bulk read')
            else:
                bulk = local.getSubhaloesStellarAssembly(simName, snapnum)
                row = local.getSubhaloStellarAssembly(simName, snapnum, 0)
                if row['StellarMassTotal'] != bulk['StellarMassTotal'][0]:
                    raise AssertionError('the row differs from the bulk read')
        except OSError:
            continue  # the file is still being downloaded
        checked += 1
    print(f'  real catalogs read: {checked} of {len(REAL_CASES)}')


if __name__ == '__main__':
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith('test_') and callable(obj)]
    for check in checks:
        check()
        print(f'PASS {check.__name__}')
    print(f'{len(checks)} checks passed')
