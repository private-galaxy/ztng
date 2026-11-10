# Data access

`ztng.accessdata` gives access to the IllustrisTNG data. It tries to read the
local files of the data release first; when a file is missing, it downloads the
result from the web-based API and caches it, so that the next query is answered
directly without any further web request.

Most users only need `auto`.

## Using auto

```python
from ztng.accessdata import auto

header = auto.getSnapHeader('TNG100-1', 99)
header['Redshift'], header['BoxSize'], header['HubbleParam']
# (2.22e-16, 75000.0, 0.6774)

auto.getSnapRedshift('TNG100-1', 99)
# 2.22e-16
auto.getSnapLookbackTime('TNG100-1', 99)
# 0.0                                   # Gyr
```

`getSubhaloSnap` returns the requested fields of one particle type, together
with a `count` entry that holds the number of particles:

```python
stars = auto.getSubhaloSnap('TNG100-1', 57, 342386, 'stars',
                            ['Masses', 'Coordinates'])
sorted(stars)
# ['Coordinates', 'Masses', 'count']
stars['count'], stars['Masses'].shape, stars['Masses'].dtype
# (44997, (44997,), dtype('float32'))
```

The particle types are `'gas'`, `'dm'`, `'stars'` and `'bhs'`; tracers cannot
be queried. An object that has no particles of the requested type returns
`{'count': 0}` and no field, which is the normal case for the stars of a dark
subhalo.

Group catalog information and merger trees are returned as plain dictionaries:

```python
info = auto.getSubhaloInfo('TNG100-1', 57, 342386)
info['SubhaloLenType']
# [71783, 141035, 0, 0, 44997, 1]

tree = auto.getSubhaloTree('TNG100-1', 57, 342386, 'sublink_mpb')
tree['SnapNum'][:3]
# array([57, 56, 55], dtype=int16)
```

The tree types are `'sublink'`, `'sublink_mpb'` and `'sublink_mdb'` for the
local files; the API also provides `'sublink_simple'`, `'lhalotree'` and
others.

## The default data layout

The files of the IllustrisTNG data release are read in the layout that
`illustris_python` expects:

```
~/data/simulations/
|- TNG100-1/                         # one directory per simulation
|  |- output/                        # groups_NNN/ and snapdir_NNN/
|  |- postprocessing/trees/SubLink/  # merger trees
|  '- supplementary/                 # supplementary data catalogs
|- TNG100-1-Dark/                    # same structure as TNG100-1/
'- TNG-API/                          # cache of the online queries
```

A host that stores the data elsewhere should keep this structure and link it
into place:

```bash
mkdir -p ~/data/simulations
ln -s /path/to/IllustrisTNG/TNG100-1 ~/data/simulations/TNG100-1
ln -s /path/to/IllustrisTNG/TNG100-1-Dark ~/data/simulations/TNG100-1-Dark
```

`TNG-API/` is not created automatically. Create it before the first query:

```bash
mkdir -p ~/data/simulations/TNG-API
```

On macOS the default file system is case insensitive, so `~/data/simulations/`
and `~/Data/Simulations/` are the same directory.

## The API key

The API requests carry a personal key in a header. Register at
<https://www.tng-project.org/users/register/> and export the key:

```bash
export TNG_API_KEY="your key"
```

Only downloads need it: a query that is answered from the local files or from
the cache never reads the variable.

## The three backends

| module | source | needs the API key | needs the data release files |
| --- | --- | --- | --- |
| `local` | the files of the data release | no | yes |
| `online` | the web-based API, cached locally | yes | no |
| `auto` | the local files, the API as a fallback | only when it falls back | no |

`local` is a thin wrapper around `illustris_python`. It downloads nothing and
raises `FileNotFoundError` when a file is missing.

`online` queries the API and caches every response under `TNG-API/`. It needs
both the API key and a writable cache directory.

`auto` tries `local` first and calls `online` when a local copy is missing,
which is why the examples above use it.

## Caching

Everything that `online` fetches is stored once and then reused, so a repeated
query is answered without any request. Two kinds of cached data behave
differently:

- Metadata, group catalog information and merger trees are saved as a whole:
  the first query downloads the file, every later one read it back.
- Snapshot data is saved per object and per particle type, and is extended
  field by field. Asking for a field that is not cached yet downloads that
  field and merges it into the same file; asking again is free.

Two details are worth knowing:

- Passing an empty field list, `fields=[]`, means *all* the fields of that
  particle type. The whole cutout is then downloaded and a warning says so.
- Deleting a cached file forces a fresh download the next time it is needed.

The cache lives in `~/data/simulations/TNG-API/`, one file per object or per metadata response, as named by
`ztng.accessdata.utils.getSavedFileName`.
