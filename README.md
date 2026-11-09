# ztng
My tools for the IllustrisTNG simulations.

Documentation: [English](docs/accessdata.md) | [中文](docs/accessdata.zh-CN.md)

## Installation

Python 3.11 or newer is required. Clone the repository and install it in place:

```bash
git clone git@github.com:private-galaxy/ztng.git
cd ztng
pip install -e .
```

Install the extra dependencies to run the checks:

```bash
pip install -e '.[dev]'
pytest
```
