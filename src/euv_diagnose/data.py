"""Read released intensity measurements without clipping negative observations."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def read_observations(path: str | Path, grid, *, scan=None):
    """Match exact released wavelengths/angles; never interpolate silently.

    scan is zero-based for a multi-acquisition file. Output follows grid order.
    """
    d = loadmat(path, simplify_cells=True)
    values = np.asarray(d["R"], dtype=float)
    if values.ndim == 3:
        if scan is None or not 0 <= scan < values.shape[2]:
            raise ValueError("An explicit valid scan index is required")
        values = values[:, :, scan]
    elif scan is not None:
        raise ValueError("Single-acquisition file does not accept scan index")
    wl = np.asarray(d["lambda"])
    angle = np.asarray(d["theta"])
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 2 or not np.isfinite(grid).all():
        raise ValueError("Grid must contain finite wavelength/angle pairs")
    iw = np.isclose(grid[:, 0, None], wl[None, :], rtol=0, atol=1e-9)
    ia = np.isclose(grid[:, 1, None], angle[None, :], rtol=0, atol=1e-9)
    if not ((iw.sum(axis=1) == 1).all() and (ia.sum(axis=1) == 1).all()):
        raise ValueError("Requested grid absent or duplicated in measured axes")
    result = values[iw.argmax(axis=1), ia.argmax(axis=1)]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite measured intensity")
    return result
