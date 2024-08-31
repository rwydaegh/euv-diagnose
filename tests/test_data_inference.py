from pathlib import Path

import numpy as np
import pytest

from euv_diagnose.data import read_observations
from euv_diagnose.inference import fit, predict
from euv_diagnose.optics import ReleasedModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research/sources/sherwin"


def test_negative_observations_preserved_and_missing_grid_rejected():
    m = ReleasedModel.load(SRC / "FIlm models/Reflectivityapp_workspace_fit131.mat")
    p = SRC / "Reflectivityapp/TaN_131/Data/Reflectivityapp_workspace_2019_08_21_Abs.mat"
    y = read_observations(p, m.grid)
    assert (y < 0).sum() == 1
    with pytest.raises(ValueError):
        read_observations(p, [[13.501, 2]])


def test_synthetic_recovery_and_holdout_isolation():
    m = ReleasedModel.load(SRC / "FIlm models/Reflectivityapp_workspace_fit131.mat")
    truth = np.array([0.8, 1.04, 0.08, 0.0004])
    y = predict(m, truth)
    train = m.grid[:, 1] <= 5
    first = fit(m, y, train, starts=1)
    altered = y.copy()
    altered[~train] += 10
    second = fit(m, altered, train, starts=1)
    assert first.success and second.success
    np.testing.assert_allclose(first.parameters, truth, atol=1e-6)
    np.testing.assert_array_equal(first.parameters, second.parameters)
