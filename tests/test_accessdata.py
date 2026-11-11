"""
Runnable checks of ztng.accessdata.

Usage:
    python tests/test_accessdata.py
    pytest tests/test_accessdata.py

The checks need neither network access nor the real data cache: the saved
data path is redirected to a temporary directory, and the HTTP helper is
replaced by a stub that fails the check if it is ever called.
"""

import os
import re
import sys
import tempfile
import warnings

import h5py
import numpy as np
import requests
from tenacity import wait_none

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ztng.accessdata import local, online, utils  # noqa: E402


def _blocked(*args, **kwargs):
    """ Fail the check when the code under test tries to use the network. """
    raise AssertionError(f'unexpected HTTP request: {args} {kwargs}')


class isolated:
    """ Redirect the cache and the network helper to safe substitutes. """

    def __enter__(self):
        self.original = (utils.getSavedDataPath, utils.get,
                         online.get, online.getSimBasePath,
                         online.getSubhaloInfo)
        self.tempdir = tempfile.mkdtemp()
        utils.getSavedDataPath = lambda: self.tempdir
        utils.get = _blocked
        online.get = _blocked
        # a base path that does not exist, so that no local simulation data
        # is read and the code falls back to the stubbed API information
        online.getSimBasePath = lambda simName: os.path.join(
            self.tempdir, 'nonexistent', 'output/')
        # no particles of any type, so that no cutout is ever requested
        online.getSubhaloInfo = lambda simName, snapnum, sfid: {
            'SubhaloLenType': [0, 0, 0, 0, 0, 0]}
        return self

    def __exit__(self, *exc):
        (utils.getSavedDataPath, utils.get, online.get,
         online.getSimBasePath, online.getSubhaloInfo) = self.original
        return False


def test_saved_file_name():
    """ The name of every saved data file is stable. """
    cases = [
        ('simulation_metadata', dict(simName='TNG100-1'),
         'tng100-1_simulation_metadata.npy'),
        ('snapshot_metadata', dict(simName='TNG100-1', snapnum=99),
         'tng100-1_snapshot_metadata_99.npy'),
        ('snapshot_metadata', dict(simName='TNG100-1', snapnum=7),
         'tng100-1_snapshot_metadata_07.npy'),
        ('subhalo_detail', dict(simName='TNG100-1', snapnum=57, sfid=342386),
         'tng100-1_subhalo_detail_57_342386.npy'),
        ('subhalo_information', dict(simName='TNG100-1', snapnum=57, sfid=1),
         'tng100-1_subhalo_information_57_1.npy'),
        ('subhalo_snapshotdata', dict(simName='TNG100-1', snapnum=57, sfid=1),
         'tng100-1_subhalo_snapshotdata_57_1.hdf5'),
        ('subhalo_tree', dict(simName='TNG100-1', snapnum=57, sfid=1,
                              treeType='sublink_mpb'),
         'tng100-1_subhalo_tree_sublink_mpb_57_1.hdf5'),
        ('subhalo_tree', dict(simName='TNG100-1', snapnum=57, sfid=1,
                              treeType='sublink_simple'),
         'tng100-1_subhalo_tree_sublink_simple_57_1.npy'),
        ('halo_information', dict(simName='TNG100-1', snapnum=99, fofid=2),
         'tng100-1_halo_information_99_2.npy'),
        ('halo_snapshotdata', dict(simName='TNG100-1', snapnum=99, fofid=2),
         'tng100-1_halo_snapshotdata_99_2.hdf5'),
    ]
    with isolated():
        for file_type, kwargs, expected in cases:
            got = utils.getSavedFileName(file_type=file_type, **kwargs)
            assert os.path.basename(got) == expected, \
                f'{file_type}: {got} != {expected}'
            assert os.path.dirname(got) == utils.getSavedDataPath(), got

        invalid = [
            dict(file_type='nonsense', simName='TNG100-1'),
            dict(file_type='simulation_metadata', simName='TNG100-1', foo=1),
        ]
        for kwargs in invalid:
            try:
                utils.getSavedFileName(**kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{kwargs} should raise ValueError')

        # a missing keyword argument names itself instead of raising KeyError
        incomplete = [
            dict(file_type='subhalo_detail', simName='TNG100-1'),
            dict(file_type='subhalo_tree', simName='TNG100-1', snapnum=1,
                 sfid=2),
            dict(file_type='halo_information'),
        ]
        for kwargs in incomplete:
            try:
                utils.getSavedFileName(**kwargs)
            except ValueError as err:
                assert 'needs the keyword argument' in str(err), str(err)
            else:
                raise AssertionError(f'{kwargs} should raise ValueError')


def test_data_path_is_not_created():
    """ A missing data directory is reported, never created. """
    original = utils.getSavedDataPath
    missing = os.path.join(tempfile.mkdtemp(), 'nonexistent', 'TNG-API')
    utils.getSavedDataPath = lambda: missing
    try:
        try:
            utils.getSavedFileName(file_type='simulation_metadata',
                                   simName='TNG100-1')
        except RuntimeError as err:
            assert 'mkdir -p' in str(err), str(err)
            assert missing in str(err), str(err)
        else:
            raise AssertionError('a missing data directory should raise '
                                 'RuntimeError')
        assert not os.path.exists(missing), 'the directory must not be created'
    finally:
        utils.getSavedDataPath = original


def test_default_paths():
    """ The default paths use the lower case directory names. """
    simPath = utils.getSimBasePath('TNG100-1')
    expected = os.path.join(os.path.expanduser('~'), 'data', 'simulations',
                            'TNG100-1', 'output') + os.sep
    assert simPath == expected, simPath

    dataPath = utils.getSavedDataPath()
    expected = os.path.join(os.path.expanduser('~'), 'data', 'simulations',
                            'TNG-API') + os.sep
    assert dataPath == expected, dataPath


def test_allowed_partType():
    """ Valid particle types map to their names, invalid ones raise. """
    valid = {
        'gas': 'gas', 0: 'gas', 'Gas': 'gas', '0': 'gas', 'cells': 'gas',
        'dm': 'dm', 1: 'dm', 'darkmatter': 'dm',
        'stars': 'stars', 4: 'stars', 'stellar': 'stars', 'wind': 'stars',
        'bhs': 'bhs', 5: 'bhs', 'blackhole': 'bhs',
    }
    for partType, expected in valid.items():
        got = utils.allowed_partType(partType)
        assert got == expected, f'{partType!r}: {got} != {expected}'

    for partType in ['tracer', 'tracers', 3, 2, 'dmlowres', 'foo', '']:
        try:
            utils.allowed_partType(partType)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{partType!r} should raise ValueError')


def test_local_partType_is_checked():
    """ The local reader accepts every type illustris_python knows. """
    for partType in ['gas', 'dm', 'tracers', 'stars', 'wind', 'bhs', 4,
                     'Stars']:
        local._checkPartType(partType)

    for partType in ['foo', '', None, 'all']:
        try:
            local._checkPartType(partType)
        except ValueError as err:
            assert repr(partType) in str(err), str(err)
        else:
            raise AssertionError(f'{partType!r} should raise ValueError')


def test_partStandardName():
    """ Particle types map to their HDF5 group names. """
    cases = {'gas': 'PartType0', 'dm': 'PartType1', 'tracer': 'PartType3',
             'stars': 'PartType4', 'bhs': 'PartType5', 4: 'PartType4',
             '4': 'PartType4'}
    for partType, expected in cases.items():
        got = utils.partStandardName(partType)
        assert got == expected, f'{partType!r}: {got} != {expected}'


def test_api_key():
    """ The API key is read from the environment only. """
    original = os.environ.pop('TNG_API_KEY', None)
    try:
        try:
            utils.getApiKey()
        except RuntimeError:
            pass
        else:
            raise AssertionError('getApiKey should raise RuntimeError')
        os.environ['TNG_API_KEY'] = 'dummy-key'
        assert utils.getApiKey() == 'dummy-key'
    finally:
        os.environ.pop('TNG_API_KEY', None)
        if original is not None:
            os.environ['TNG_API_KEY'] = original

    # no source file may contain a hard coded API key
    srcDir = os.path.join(os.path.dirname(__file__), '..', 'src')
    pattern = re.compile(r"""['"]api-key['"]\s*:\s*['"][0-9a-zA-Z]{16,}""")
    for root, _, files in os.walk(srcDir):
        for name in files:
            if not name.endswith('.py'):
                continue
            path = os.path.join(root, name)
            with open(path, encoding='utf-8') as f:
                assert not pattern.search(f.read()), \
                    f'{path} contains a hard coded API key'


def test_meta_is_cached():
    """ A queried metadata is downloaded once and then read locally. """
    calls = []

    def fakeGet(path, params=None):
        calls.append(path)
        return {'redshift': 0.0}

    with isolated():
        online.get = fakeGet
        assert online.getSnapMeta('TNG100-1', 42) == {'redshift': 0.0}
        assert len(calls) == 1, calls
        assert online.getSnapMeta('TNG100-1', 42) == {'redshift': 0.0}
        assert len(calls) == 1, calls

        filename = utils.getSavedFileName(file_type='snapshot_metadata',
                                          simName='TNG100-1', snapnum=42)
        assert os.path.exists(filename)
        prefix = os.path.basename(filename)
        leftovers = [name for name in os.listdir(os.path.dirname(filename))
                     if name.startswith(prefix)]
        assert leftovers == [prefix], leftovers


def test_tree_is_cached():
    """ A JSON merger tree is downloaded once and then read locally. """
    # the shape of sublink/simple.json, as returned by the TNG API
    payload = {'Main': [[99, 7], [98, 5], [97, 3]], 'Mergers': []}
    calls = []

    def fakeGet(path, params=None):
        calls.append(path)
        if path.endswith('simple.json'):
            return payload
        return {'trees': {'sublink_simple': path + 'sublink/simple.json'}}

    with isolated():
        online.get = fakeGet
        assert online.getSubhaloTree(
            'TNG100-1', 99, 7, 'sublink_simple') == payload
        numberOfCalls = len(calls)
        assert online.getSubhaloTree(
            'TNG100-1', 99, 7, 'sublink_simple') == payload
        assert len(calls) == numberOfCalls, calls

    # an unknown tree type names itself instead of raising KeyError
    with isolated():
        online.get = fakeGet
        try:
            online.getSubhaloTree('TNG100-1', 99, 7, 'bogus')
        except ValueError as err:
            assert 'bogus' in str(err), str(err)
        else:
            raise AssertionError('an unknown treeType should raise ValueError')


def test_retry_policy():
    """ A 4xx response is not retried, a 5xx one is. """
    calls = []

    class FakeResponse:
        def __init__(self, status):
            self.status_code = status
            self.reason = 'fake'
            self.headers = {'content-type': 'application/json'}
            self.content = b'{}'

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f'{self.status_code} fake',
                                         response=self)

    def makeGet(status):
        def fakeGet(path, params=None, headers=None, timeout=None):
            calls.append(status)
            return FakeResponse(status)
        return fakeGet

    originalGet = utils.requests.get
    originalWait = utils.get.retry.wait
    originalKey = os.environ.get('TNG_API_KEY')
    utils.get.retry.wait = wait_none()
    os.environ['TNG_API_KEY'] = 'dummy-key'
    try:
        # 4xx: a client error must fail at once
        utils.requests.get = makeGet(400)
        try:
            utils.get('https://example.invalid/api/')
        except requests.HTTPError:
            pass
        else:
            raise AssertionError('a 400 response should raise HTTPError')
        assert len(calls) == 1, calls

        # 5xx: the request is retried, up to the attempt limit
        calls.clear()
        utils.requests.get = makeGet(503)
        try:
            utils.get('https://example.invalid/api/')
        except requests.HTTPError:
            pass
        else:
            raise AssertionError('a 503 response should raise HTTPError')
        assert len(calls) == 6, calls

        # 429 is retried as well, because it is the rate limit
        calls.clear()
        utils.requests.get = makeGet(429)
        try:
            utils.get('https://example.invalid/api/')
        except requests.HTTPError:
            pass
        else:
            raise AssertionError('a 429 response should raise HTTPError')
        assert len(calls) == 6, calls
    finally:
        utils.requests.get = originalGet
        utils.get.retry.wait = originalWait
        if originalKey is None:
            os.environ.pop('TNG_API_KEY', None)
        else:
            os.environ['TNG_API_KEY'] = originalKey


def _prepare_cache(ids, partType, fields):
    """ Create a cached snapshot file that already holds some fields. """
    online._loadSnap('subhalo_snapshotdata', **ids)  # creates empty groups
    filename = utils.getSavedFileName(
        file_type='subhalo_snapshotdata', **ids)
    with h5py.File(filename, 'r+') as f:
        for field, data in fields.items():
            f[partType][field] = data
    return filename


def test_snap_reads_the_cache():
    """ Cached fields are returned without any HTTP request. """
    rng = np.random.default_rng(20260324)
    coords = rng.random((5, 3))
    masses = rng.random(5)
    ids = dict(simName='TNG100-1', snapnum=99, sfid=42)

    with isolated():
        _prepare_cache(ids, 'PartType4',
                       dict(Coordinates=coords, Masses=masses))

        # a subset of the cached fields
        result = online.getSubhaloSnap('TNG100-1', 99, 42, 'stars',
                                       ['Masses'])
        assert sorted(result) == ['Masses', 'count'], sorted(result)
        assert result['count'] == 5, result['count']
        assert np.array_equal(result['Masses'], masses)

        # the order of the requested fields does not change 'count'
        result = online.getSubhaloSnap('TNG100-1', 99, 42, 'stars',
                                       ['Masses', 'Coordinates'])
        assert result['count'] == 5, result['count']
        assert np.array_equal(result['Coordinates'], coords)


def test_snap_empty_fields_means_all():
    """ An empty field list downloads and returns every field. """
    rng = np.random.default_rng(7)
    coords = rng.random((4, 3))
    masses = rng.random(4)

    with isolated():
        requests = []

        def fakeGet(path, params=None):
            requests.append((path, params))
            return _fake_cutout(
                'PartType4', dict(Coordinates=coords, Masses=masses))

        online.getSubhaloInfo = lambda simName, snapnum, sfid: {
            'SubhaloLenType': [0, 0, 0, 0, 4, 0]}
        online.get = fakeGet

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = online.getSubhaloSnap('TNG100-1', 99, 43, 'stars', [])

        assert sorted(result) == ['Coordinates', 'Masses', 'count'], \
            sorted(result)
        assert result['count'] == 4, result['count']
        assert np.array_equal(result['Coordinates'], coords)

        # every field is requested from the API, without a field filter
        assert len(requests) == 1, requests
        assert requests[0][0].endswith('cutout.hdf5'), requests[0][0]
        assert requests[0][1] is None, requests[0][1]

        # the user is warned that the whole cutout is downloaded
        assert len(caught) == 1, [str(w.message) for w in caught]
        assert 'all stars fields' in str(caught[0].message), \
            str(caught[0].message)


class _FakeResponse:
    """ Stand-in for a requests response that carries binary content. """

    def __init__(self, content):
        self.content = content


def _fake_cutout(partType, fields):
    """ Build the bytes of a cutout file, as the API would return them. """
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
        name = tmp.name
    try:
        with h5py.File(name, 'w') as f:
            grp = f.create_group(partType)
            for field, data in fields.items():
                grp[field] = data
        with open(name, 'rb') as f:
            return _FakeResponse(f.read())
    finally:
        os.remove(name)


def test_snap_missing_field_raises():
    """ A requested field that the API does not return raises KeyError. """
    ids = dict(simName='TNG100-1', snapnum=99, sfid=44)
    rng = np.random.default_rng(11)

    with isolated():
        _prepare_cache(ids, 'PartType4', dict(Masses=rng.random(3)))
        online.getSubhaloInfo = lambda simName, snapnum, sfid: {
            'SubhaloLenType': [0, 0, 0, 0, 3, 0]}
        online.get = lambda path, params=None: _fake_cutout(
            'PartType4', dict(Masses=rng.random(3)))
        try:
            online.getSubhaloSnap('TNG100-1', 99, 44, 'stars', ['Velocities'])
        except KeyError as err:
            assert 'Velocities' in str(err), str(err)
        else:
            raise AssertionError('a missing field should raise KeyError')


def test_snap_merges_new_fields():
    """ A downloaded field is merged into the cache and returned. """
    ids = dict(simName='TNG100-1', snapnum=99, sfid=46)
    rng = np.random.default_rng(3)
    masses = rng.random(3)
    velocities = rng.random((3, 3))

    with isolated():
        _prepare_cache(ids, 'PartType4', dict(Masses=masses))
        online.getSubhaloInfo = lambda simName, snapnum, sfid: {
            'SubhaloLenType': [0, 0, 0, 0, 3, 0]}
        online.get = lambda path, params=None: _fake_cutout(
            'PartType4', dict(Masses=masses, Velocities=velocities))

        result = online.getSubhaloSnap('TNG100-1', 99, 46, 'stars',
                                       ['Velocities'])
        assert result['count'] == 3, result['count']
        assert np.array_equal(result['Velocities'], velocities)

        # the downloaded field is now part of the cache
        cached = online.loadSubhaloSnap('TNG100-1', 99, 46)['PartType4']
        assert sorted(cached) == ['Masses', 'Velocities'], sorted(cached)
        assert np.array_equal(cached['Masses'], masses)

        # no temporary file is left behind
        filename = utils.getSavedFileName(
            file_type='subhalo_snapshotdata', **ids)
        leftovers = [name for name in os.listdir(os.path.dirname(filename))
                     if '.api.' in name or '.new.' in name]
        assert leftovers == [], leftovers


def test_snap_empty_object():
    """ A zero-particle object still produces 'count': 0. """
    ids = dict(simName='TNG100-1', snapnum=99, sfid=45)

    with isolated():
        online._loadSnap('subhalo_snapshotdata', **ids)
        result = online.getSubhaloSnap('TNG100-1', 99, 45, 'gas', ['Masses'])
        assert result == {'count': 0}, result

        # an empty field list on a zero-particle object is also harmless
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = online.getSubhaloSnap('TNG100-1', 99, 45, 'gas', [])
        assert result == {'count': 0}, result
        assert len(caught) == 0, [str(w.message) for w in caught]

        # the five particle type groups are created up front
        filename = utils.getSavedFileName(
            file_type='subhalo_snapshotdata', **ids)
        with h5py.File(filename, 'r') as f:
            assert sorted(f.keys()) == ['PartType0', 'PartType1', 'PartType3',
                                        'PartType4', 'PartType5'], \
                sorted(f.keys())

        # no temporary file is left behind
        leftovers = [name for name in os.listdir(os.path.dirname(filename))
                     if '.api.' in name or '.new.' in name]
        assert leftovers == [], leftovers


def test_snap_rejects_unknown_kind():
    """ An unknown object kind raises ValueError instead of guessing. """
    with isolated():
        try:
            online._getInfo('nonsense', 'TNG100-1', 99, 42)
        except ValueError:
            pass
        else:
            raise AssertionError('_getInfo should raise ValueError')
        try:
            online._getSnap('nonsense', 'TNG100-1', 99, 42, 'stars',
                            ['Masses'])
        except ValueError:
            pass
        else:
            raise AssertionError('_getSnap should raise ValueError')


if __name__ == '__main__':
    checks = [obj for name, obj in sorted(globals().items())
              if name.startswith('test_') and callable(obj)]
    for check in checks:
        check()
        print(f'PASS {check.__name__}')
    print(f'{len(checks)} checks passed')
