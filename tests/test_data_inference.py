from pathlib import Path

import numpy as np
import pytest

from euv_diagnose.data import read_observations
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


