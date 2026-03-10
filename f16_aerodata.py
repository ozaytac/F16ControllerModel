"""F-16 aerodynamic lookup tables — parsed from F16aerodata.m at runtime.

Usage
-----
from f16_aerodata import get_f16_aerodata
aero = get_f16_aerodata()   # SimpleNamespace of numpy arrays
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_M_FILE = Path(__file__).with_name("F16aerodata.m")


def _parse_matrix(body: str) -> np.ndarray:
    """Convert a MATLAB matrix body (text between '[' and ']') to ndarray."""
    body = re.sub(r"%[^\n]*", "", body)           # strip inline comments
    rows = [r.strip() for r in body.split(";") if r.strip()]
    data = [[float(v) for v in row.split()] for row in rows]
    return np.squeeze(np.array(data, dtype=np.float64))


def get_f16_aerodata(path: Path | str = _M_FILE) -> SimpleNamespace:
    """Parse *F16aerodata.m* and return a :class:`~types.SimpleNamespace`.

    Every ``f16data.NAME`` assignment becomes an attribute.  Three-dimensional
    tables (``f16data.NAME(:,:,k)``) are stacked along ``axis=2`` in MATLAB
    slice order.

    Parameters
    ----------
    path : path-like, optional
        Location of ``F16aerodata.m``.  Defaults to the copy alongside this
        module.

    Returns
    -------
    types.SimpleNamespace
        Attributes are numpy ``float64`` arrays whose shapes mirror the
        original MATLAB variables.
    """
    src = Path(path).read_text()

    ns = SimpleNamespace()
    slices: dict[str, dict[int, np.ndarray]] = {}

    # 3-D slices: f16data.NAME(:,:,k) = [...];
    for m in re.finditer(
        r"f16data\.(\w+)\(:,:,(\d+)\)\s*=\s*\[([^\]]+)\]",
        src, re.DOTALL,
    ):
        name, k, body = m.group(1), int(m.group(2)), m.group(3)
        slices.setdefault(name, {})[k] = _parse_matrix(body)

    # Scalar / 1-D / 2-D: f16data.NAME = [...];
    for m in re.finditer(
        r"f16data\.(\w+)\s*=\s*\[([^\]]+)\]",
        src, re.DOTALL,
    ):
        name, body = m.group(1), m.group(2)
        if name not in slices:          # don't clobber 3-D tables
            setattr(ns, name, _parse_matrix(body))

    # Stack 3-D slices along axis=2 in MATLAB index order
    for name, sdict in slices.items():
        ordered = [sdict[k] for k in sorted(sdict)]
        setattr(ns, name, np.stack(ordered, axis=2))

    return ns
