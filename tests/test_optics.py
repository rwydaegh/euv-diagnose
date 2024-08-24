from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from euv_diagnose.optics import ReleasedModel, transfer_matrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("tag", ["131", "073", "074"])
def test_original_complex_amplitude(tag):
    m = ReleasedModel.load(
        ROOT / f"research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit{tag}.mat"
    )
    d = loadmat(ROOT / f"research/results/sherwin-replay-{tag}.mat", simplify_cells=True)
    np.testing.assert_allclose(
        m.amplitude(), np.column_stack([d["r_abs"], d["r_etch"]]), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("tag", ["131", "073", "074"])
@pytest.mark.parametrize("pol", ["s", "p"])
def test_perturbed_original_complex_amplitude(tag, pol):
    m = ReleasedModel.load(
        ROOT / f"research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit{tag}.mat"
    )
    m = replace(m, polarization=pol)
    x = m.x.copy()
    x[m.indices["thick"][0]] += 0.03
    x[m.indices["thick"][-1]] += 0.07
    x[m.indices["rough"]] += 0.02
    d = loadmat(ROOT / f"research/results/optics-reference-{tag}-{pol}.mat", simplify_cells=True)
    np.testing.assert_allclose(
        m.amplitude(x, angle_shift=0.03), np.column_stack([d["a"], d["b"]]), rtol=1e-10, atol=1e-12
    )














