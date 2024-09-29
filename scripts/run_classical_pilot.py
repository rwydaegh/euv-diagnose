import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from euv_diagnose.data import read_observations
from euv_diagnose.inference import LOWER, PARAMETERS, UPPER, fit, predict
from euv_diagnose.optics import ReleasedModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research/sources/sherwin"
OUT = ROOT / "research/results"
model = ReleasedModel.load(SRC / "FIlm models/Reflectivityapp_workspace_fit131.mat")
path = SRC / "Reflectivityapp/TaN_131/Data/Reflectivityapp_workspace_2019_11_22_AbsML.mat"

observed = read_observations(path, model.grid, scan=21)
train = model.grid[:, 1] <= 5
test = ~train
truth = np.array([0.8, 1.04, 0.08, 0.0004])
synthetic = predict(model, truth)
report = {
    "scope": "Exploratory repeat scan 22, multilayer only, initial stack fixed to saved fit. Not causal identification, calibrated uncertainty, or a blind final evaluation.",
    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "code_files_sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(), *sorted((ROOT / "src/euv_diagnose").glob("*.py"))]
    },
    "code_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    "parameters": list(PARAMETERS),
    "bounds": {"lower": LOWER.tolist(), "upper": UPPER.tolist()},
    "train_angles_deg": [2, 3, 4, 5],
    "test_angles_deg": [6, 7, 8],
    "weighting": "Equal absolute intensity residuals, no inferred noise covariance.",
    "random_seed": 42,
    "starts": 3,
    "fits": [],
}
t0 = time.perf_counter()
for _ in range(20):
    model.intensity()
report["mean_forward_seconds_20_calls"] = (time.perf_counter() - t0) / 20
s = fit(model, synthetic, train, starts=2)
report["noiseless_synthetic_recovery"] = {
    "truth": truth.tolist(),
    "recovered": s.parameters.tolist(),
    "max_abs_prediction_error": float(np.max(abs(s.prediction - synthetic))),
    "seconds": s.seconds,
    "success": s.success,
}
assert np.max(abs(s.prediction - synthetic)) < 1e-7
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
selected = model.grid[:, 1] == 7
axes[0].plot(model.grid[selected, 0], observed[selected], "k.", ms=4, label="Measured: held-out 7°")
for mode in ["carbon", "calibration", "joint"]:
    r = fit(model, observed, train, mode=mode)
    item = asdict(r)
    item.pop("prediction")
    item["parameters"] = r.parameters.tolist()
    item.update(
        mode=mode, test_rmse=float(np.sqrt(np.mean((r.prediction[test] - observed[test]) ** 2)))
    )
    report["fits"].append(item)
    axes[0].plot(model.grid[selected, 0], r.prediction[selected], label=mode)
    axes[1].plot(model.grid[selected, 0], r.prediction[selected] - observed[selected], label=mode)
axes[0].set(
    xlabel="Wavelength (nm)", ylabel="Reflectivity", title="Conditional-stack classical fits"
)
axes[1].set(
    xlabel="Wavelength (nm)", ylabel="Prediction − measurement", title="Unfitted angle: residuals"
)
axes[1].axhline(0, color="gray", lw=0.7)
axes[0].legend()
axes[1].legend()
fig.savefig(OUT / "classical-pilot.png", dpi=170)


report["sensitivity_excluding_2deg"] = {
    "reason": "Strong anomalous spectral shape at labeled 2 degrees; cause unresolved. Post-hoc sensitivity, not corrected data.",
    "fits": [],
}
for mode in ["carbon", "calibration", "joint"]:
    r = fit(model, observed, train & (model.grid[:, 1] > 2), mode=mode)
    item = asdict(r)
    item.pop("prediction")
    item["parameters"] = r.parameters.tolist()
    item.update(
        mode=mode, test_rmse=float(np.sqrt(np.mean((r.prediction[test] - observed[test]) ** 2)))
    )
    report["sensitivity_excluding_2deg"]["fits"].append(item)
(OUT / "classical-pilot.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
