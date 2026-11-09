# 数据访问

[English](accessdata.md) | **中文**

`ztng.accessdata` 用于读取 IllustrisTNG 数据。它首先尝试读取 data release 的
本地文件；当文件不存在时，就从在线 API 下载结果并缓存，因此下一次查询直接得到
答复，不会再发出网络请求。

大多数时候只需用到 `auto`。

## 使用 auto

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

`getSubhaloSnap` 返回某一 particle type 的指定字段，并附带一个 `count`，记录
粒子数：

```python
stars = auto.getSubhaloSnap('TNG100-1', 57, 342386, 'stars',
                            ['Masses', 'Coordinates'])
sorted(stars)
# ['Coordinates', 'Masses', 'count']
stars['count'], stars['Masses'].shape, stars['Masses'].dtype
# (44997, (44997,), dtype('float32'))
```

可查询的 particle type 是 `'gas'`、`'dm'`、`'stars'` 和 `'bhs'`；tracer 不能
查询。若对象没有该 particle type 的粒子，则返回 `{'count': 0}` 且不含任何
字段——暗 subhalo 的 stars 就是这种情况。

group catalog 信息与 merger trees 以普通字典返回：

```python
info = auto.getSubhaloInfo('TNG100-1', 57, 342386)
info['SubhaloLenType']
# [71783, 141035, 0, 0, 44997, 1]

tree = auto.getSubhaloTree('TNG100-1', 57, 342386, 'sublink_mpb')
tree['SnapNum'][:3]
# array([57, 56, 55], dtype=int16)
```

本地文件支持的 tree 类型是 `'sublink'`、`'sublink_mpb'` 和 `'sublink_mdb'`；
API 还提供 `'sublink_simple'`、`'lhalotree'` 等。

## 默认的数据目录结构

data release 的文件按 `illustris_python` 期望的结构读取：

```
~/data/simulations/
|- TNG100-1/                         # 每个模拟一个目录
|  |- output/                        # groups_NNN/ and snapdir_NNN/
|  |- postprocessing/trees/SubLink/  # merger trees
|  '- supplementary/                 # supplementary data catalogs
|- TNG100-1-Dark/                    # 结构与 TNG100-1/ 相同
'- TNG-API/                          # 在线查询的缓存
```

若数据存放在别处，应保持这个结构并链接进来：

```bash
mkdir -p ~/data/simulations
ln -s /path/to/IllustrisTNG/TNG100-1 ~/data/simulations/TNG100-1
ln -s /path/to/IllustrisTNG/TNG100-1-Dark ~/data/simulations/TNG100-1-Dark
```

`TNG-API/` 不会被自动创建。首次查询前请自行创建：

```bash
mkdir -p ~/data/simulations/TNG-API
```

macOS 默认的文件系统不区分大小写，因此 `~/data/simulations/` 与
`~/Data/Simulations/` 是同一个目录。

## API key

API 请求在 header 中携带个人 key。在
<https://www.tng-project.org/users/register/> 注册，然后导出：

```bash
export TNG_API_KEY="your key"
```

只有下载需要它：从本地文件或缓存得到答案的查询不会读取这个变量。

## 三个后端

| 模块 | 数据来源 | 需要 API key | 需要 data release 文件 |
| --- | --- | --- | --- |
| `local` | data release 的文件 | 不需要 | 需要 |
| `online` | 在线 API，本地缓存 | 需要 | 不需要 |
| `auto` | 先读本地文件，缺失时走 API | 仅在回退到 API 时需要 | 不需要 |

`local` 是 `illustris_python` 的薄封装。它不下载任何东西，文件缺失时抛
`FileNotFoundError`。

`online` 查询 API，并把每个响应缓存在 `TNG-API/` 下。它需要 API key 与一个
可写的缓存目录。

`auto` 先试 `local`，本地没有副本时调用 `online`，上面的示例用的就是它。

## 缓存

`online` 取回的一切都会保存一次然后复用，因此重复查询不会发出任何请求。两类
缓存的行为不同：

- metadata、group catalog 信息与 merger trees 是整份保存：首次查询下载文件，
  之后每次读回。
- snapshot data 按对象、按 particle type 保存，并按字段扩展。请求尚未缓存的
  字段时，只下载该字段并合并进同一个文件；再次请求则无需下载。

两个细节值得注意：

- 传入空的字段列表 `fields=[]` 表示该 particle type 的*全部*字段。此时会下载
  整份 cutout，并给出警告。
- 删除某个缓存文件，下次需要时会重新下载。

缓存位于 `~/data/simulations/TNG-API/`，每个对象或每个 metadata 响应一个文件，
命名由 `ztng.accessdata.utils.getSavedFileName` 决定。
