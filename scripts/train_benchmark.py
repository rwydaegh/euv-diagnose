import argparse
import hashlib
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from euv_diagnose.learning import phase_error
from euv_diagnose.optics import ReleasedModel

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit131.mat"


def setup():
    model = replace(ReleasedModel.load(SOURCE), periodic_convention="continuous")
    table = loadmat(SOURCE, simplify_cells=True)["table"]
    lower, upper = table[:, 3].astype(float), table[:, 6].astype(float)
    ids = np.flatnonzero(upper > lower)
    material = np.array([table[k, 1] == "Concentration" for k in ids])
    rows = np.flatnonzero(
        np.isin(model.grid[:, 1], [2, 5, 8])
        & np.isclose((model.grid[:, 0] - 12.5) / 0.1, np.round((model.grid[:, 0] - 12.5) / 0.1))
    )
    reference = np.flatnonzero(np.isclose(model.grid[:, 0], 13.5) & (model.grid[:, 1] == 6))[0]
    rows = np.r_[rows, reference]
    model = replace(model, grid=model.grid[rows], factors=model.factors[rows])
    amplitude = model.amplitude()
    phase_zero = float(np.angle(amplitude[-1, 0] / amplitude[-1, 1], deg=True))
    return model, ids, material, lower[ids], upper[ids], phase_zero


def bounds(model, ids, material, lower, upper, fraction=0.03):
    low, high = lower.copy(), upper.copy()
    low[material] = np.maximum(low[material], model.x[ids[material]] * (1 - fraction))
    high[material] = np.minimum(high[material], model.x[ids[material]] * (1 + fraction))
    return low, high










