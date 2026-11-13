"""
Paths, URLs and HTTP requests for the IllustrisTNG data.

The default data layout is:

  ~/data/simulations/
  |- TNG100-1/                         # one directory per simulation
  |  |- output/                        # groups_NNN/ and snapdir_NNN/
  |  |- postprocessing/trees/SubLink/  # merger trees
  |  '- supplementary/                 # supplementary data catalogs
  |- TNG100-1-Dark/                    # same structure as TNG100-1/
  '- TNG-API/                          # cache of the online queries

TNG-API/ is not created automatically.  Create it before the first online
query, for example with mkdir -p ~/data/simulations/TNG-API.

The directory names are written in lower case, so that the same code works on
a case sensitive file system such as Linux.  A host that stores the data
elsewhere should keep this structure and link it into place, for example:

  mkdir -p ~/data/simulations
  ln -s /path/to/IllustrisTNG/TNG100-1 ~/data/simulations/TNG100-1
  ln -s /path/to/IllustrisTNG/TNG100-1-Dark ~/data/simulations/TNG100-1-Dark

On macOS the default file system is case insensitive, so ~/data/simulations/
and ~/Data/Simulations/ are the same directory.
"""

import os
import requests
from tenacity import retry
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import stop_after_delay
from tenacity import wait_exponential_jitter
import illustris_python as il


def getSimBasePath(simName):
    """ Get the simulation data base path. """
    basePath = os.path.expanduser(f'~/data/simulations/{simName}/output/')
    return basePath


def getSavedDataPath():
    """ Get the path used to save the online data. """
    dataPath = os.path.join(
        os.path.expanduser('~'), 'data/simulations/TNG-API/')
    return dataPath


def getApiKey():
    """ Get the API key of the IllustrisTNG from the environment. """
    apiKey = os.environ.get('TNG_API_KEY')
    if not apiKey:
        raise RuntimeError(
            'the environment variable TNG_API_KEY is not set; register at '
            'https://www.tng-project.org/users/register/ and export the API '
            'key of your account as TNG_API_KEY')
    return apiKey


def getBaseUrl():
    """ Get the base url of the IllustrisTNG. """
    # NOTE: the documentation example uses 'http://'; https is used here
    # see https://www.tng-project.org/data/docs/api/
    return 'https://www.tng-project.org/api/'


def getSimUrl(simName):
    """ Get the url of the simulation. """
    baseUrl = getBaseUrl()
    return os.path.join(baseUrl, f'{simName}/')


def getSnapUrl(simName, snapnum):
    """ Get the url of the snapshot. """
    simUrl = getSimUrl(simName)
    return os.path.join(simUrl, f'snapshots/{snapnum:d}/')


def getSubhaloUrl(simName, snapnum, sfid):
    """ Get the url of the subhalo. """
    snapUrl = getSnapUrl(simName, snapnum)
    return os.path.join(snapUrl, f'subhalos/{sfid:d}/')


def getSubhaloUrls(simName, snapnum, sfid):
    """ Get the some URLs of the subhalo. """
    baseUrl = getBaseUrl()
    simUrl = getSimUrl(simName)
    snapUrl = getSnapUrl(simName, snapnum)
    subUrl = getSubhaloUrl(simName, snapnum, sfid)
    subhaloUrls = {}
    subhaloUrls['baseUrl'] = baseUrl
    subhaloUrls['simUrl'] = simUrl
    subhaloUrls['snapUrl'] = snapUrl
    subhaloUrls['subUrl'] = subUrl
    return subhaloUrls


def getHaloUrl(simName, snapnum, fofid):
    """ Get the url of the halo. """
    snapUrl = getSnapUrl(simName, snapnum)
    return os.path.join(snapUrl, f'halos/{fofid:d}/')


def _isTransientError(err):
    """ True for the HTTP failures that deserve another attempt. """
    if isinstance(err, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(err, requests.HTTPError):
        status = err.response.status_code if err.response is not None else 0
        # 429 (too many requests) or 5xx (server errors)
        return status == 429 or status >= 500
    return False


@retry(stop=(stop_after_attempt(6) | stop_after_delay(120)),
       wait=wait_exponential_jitter(initial=1, max=60), reraise=True,
       retry=retry_if_exception(_isTransientError))
def get(path, params=None, headers=None):
    """ A helper function. """
    if headers is None:
        headers = {'api-key': getApiKey()}

    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers, timeout=60)

    # raise exception if response code is not HTTP SUCCESS (200); a 429 or a
    # 5xx response is retried by _isTransientError, a 4xx response is not
    r.raise_for_status()

    content_type = r.headers.get('content-type', '')
    if content_type.startswith('application/json'):
        return r.json()  # parse json responses automatically

    return r


def partStandardName(partType):
    """ Return the standard name of the input particle type. """
    ptNum = il.util.partTypeNum(partType)
    return f'PartType{ptNum:d}'


_ALLOWED_PART_TYPES = {0: 'gas', 1: 'dm', 4: 'stars', 5: 'bhs'}


def allowed_partType(partType):
    """ {cutout_query}
    The allowed particle types are 'dm','gas','stars','bhs'. """
    try:
        ptNum = il.util.partTypeNum(partType)
    except Exception as err:
        raise ValueError(f'unknown partType {partType!r}') from err
    if ptNum not in _ALLOWED_PART_TYPES:
        raise ValueError(
            f'partType {partType!r} (PartType{ptNum}) cannot be queried '
            f'from the API; allowed: {sorted(_ALLOWED_PART_TYPES.values())}')
    return _ALLOWED_PART_TYPES[ptNum]


# every TNG run stores 100 snapshots, numbered from 0 to 99
MIN_SNAPNUM = 0
MAX_SNAPNUM = 99

# the snapshots that TNG stores with the full set of particle data
FULL_SNAPSHOTS = (2, 3, 4, 6, 8, 11, 13, 17, 21, 25,
                  33, 40, 50, 59, 67, 72, 78, 84, 91, 99)


def inFullSnap(snapnum):
    """ Whether TNG stores the full set of particle data in this snapshot. """
    return snapnum in FULL_SNAPSHOTS


_SAVED_FILE_PATTERNS = {
    'simulation_metadata': '{simName}_simulation_metadata.npy',
    'snapshot_metadata': '{simName}_snapshot_metadata_{snapnum:02d}.npy',
    'subhalo_detail': '{simName}_subhalo_detail_{snapnum:02d}_{sfid}.npy',
    'subhalo_information':
        '{simName}_subhalo_information_{snapnum:02d}_{sfid}.npy',
    'subhalo_snapshotdata':
        '{simName}_subhalo_snapshotdata_{snapnum:02d}_{sfid}.hdf5',
    'subhalo_tree':
        '{simName}_subhalo_tree_{treeType}_{snapnum:02d}_{sfid}{suffix}',
    'halo_information': '{simName}_halo_information_{snapnum:02d}_{fofid}.npy',
    'halo_snapshotdata':
        '{simName}_halo_snapshotdata_{snapnum:02d}_{fofid}.hdf5',
}

# the keyword arguments each file type needs, besides 'simName'
_SAVED_FILE_KEYS = {
    'simulation_metadata': [],
    'snapshot_metadata': ['snapnum'],
    'subhalo_detail': ['snapnum', 'sfid'],
    'subhalo_information': ['snapnum', 'sfid'],
    'subhalo_snapshotdata': ['snapnum', 'sfid'],
    'subhalo_tree': ['snapnum', 'sfid', 'treeType'],
    'halo_information': ['snapnum', 'fofid'],
    'halo_snapshotdata': ['snapnum', 'fofid'],
}

_ALLOWED_SAVED_FILE_KEYS = ['simName', 'snapnum', 'sfid', 'treeType', 'fofid']


def getSavedFileName(file_type, **kwargs):
    """ Get the file (path/)name used to save the online data. """
    if file_type not in _SAVED_FILE_PATTERNS:
        raise ValueError(
            f'file_type {file_type!r} is not allowed; '
            f'allowed: {sorted(_SAVED_FILE_PATTERNS)}')
    for kw in kwargs:
        if kw not in _ALLOWED_SAVED_FILE_KEYS:
            raise ValueError(
                f'keyword argument {kw!r} is not allowed; '
                f'allowed: {_ALLOWED_SAVED_FILE_KEYS}')
    missing = [kw for kw in ['simName'] + _SAVED_FILE_KEYS[file_type]
               if kw not in kwargs]
    if missing:
        raise ValueError(
            f'file_type {file_type!r} needs the keyword argument(s) '
            f'{missing}')

    values = dict(kwargs)
    values['simName'] = kwargs['simName'].lower()
    values['suffix'] = ('.npy' if kwargs.get('treeType') == 'sublink_simple'
                        else '.hdf5')
    filename = _SAVED_FILE_PATTERNS[file_type].format(**values)

    dataPath = getSavedDataPath()
    if not os.path.isdir(dataPath):
        raise RuntimeError(
            f'the data directory {dataPath} does not exist; create it with:'
            f'\n  mkdir -p {dataPath}')
    return os.path.join(dataPath, filename)
