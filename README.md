# ztng
[![CI](https://github.com/private-galaxy/ztng/actions/workflows/ci.yml/badge.svg)](https://github.com/private-galaxy/ztng/actions/workflows/ci.yml)

My tools for the IllustrisTNG simulations.

## Documentation

- `accessdata` — [English](docs/accessdata.md)
- `suppldata` — [English](docs/suppldata.ipynb)

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
