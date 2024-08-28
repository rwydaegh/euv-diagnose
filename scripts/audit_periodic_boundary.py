"""Compare release-compatible and continuous periodic boundaries; no refitting."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from euv_diagnose.optics import ReleasedModel, transfer_matrix

ROOT = Path(__file__).resolve().parents[1]
results = []
for tag in ["131", "073", "074"]:
    legacy = ReleasedModel.load(
        ROOT / f"research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit{tag}.mat"
    )
    continuous = replace(legacy, periodic_convention="continuous")
    a = legacy.amplitude()
    b = continuous.amplitude()
    idx = np.flatnonzero(np.isclose(legacy.grid[:, 0], 13.5) & (legacy.grid[:, 1] == 6))[0]
    results.append(
        {
            "tag": tag,
            "max_intensity_difference_same_parameters": float(
                np.max(abs(abs(a) ** 2 - abs(b) ** 2))
            ),
            "phase_difference_deg_same_parameters": float(
                np.angle((b[idx, 0] / b[idx, 1]) / (a[idx, 0] / a[idx, 1]), deg=True)
            ),
            "legacy_amplitude_relative_residual": (
                np.linalg.norm(abs(a) - legacy.observed_amplitude, axis=0)
                / np.linalg.norm(legacy.observed_amplitude, axis=0)
            ).tolist(),
            "continuous_amplitude_relative_residual_without_refit": (
                np.linalg.norm(abs(b) - legacy.observed_amplitude, axis=0)
                / np.linalg.norm(legacy.observed_amplitude, axis=0)
            ).tolist(),
        }
    )
t = [100.0, 75.0]
nk = np.array([[1.5, 2.0]])
wl = np.array([600.0])
theta = np.array([0.0])
toy = {}
for convention in ["legacy", "continuous"]:
    mat = transfer_matrix(
        t, nk, [0, 0, 0], wl, theta, periods=2, caps=0, periodic_convention=convention
    )
    toy[convention] = float(abs(mat[0, 1, 0] / mat[0, 0, 0]) ** 2)
toy["analytic"] = float((((1.5 / 2) ** 4 - 1) / ((1.5 / 2) ** 4 + 1)) ** 2)
report = {
    "scope": "Software/physics consistency audit. Same saved parameters, no refit. Lower residual alone does not establish physical validity. Legacy remains available solely for reproduction.",
    "quarter_wave_two_period_reflectivity": toy,
    "released_models": results,
}
(ROOT / "research/results/periodic-boundary-audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
