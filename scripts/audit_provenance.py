import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from euv_diagnose.data import read_observations

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research/sources/sherwin"
OUT = ROOT / "research/results"
report = {
    "models": [],
    "repeat_exclusions": "Paper-to-file scan-index mapping remains unresolved; no repeat scan silently excluded.",
}
for tag in ["131", "073", "074"]:
    d = loadmat(SRC / f"FIlm models/Reflectivityapp_workspace_fit{tag}.mat", simplify_cells=True)
    m = d["model"]
    channels = []
    for col, kind in enumerate(["Absorber", "Multilayer"]):
        cell = next(c for c in d["data_cell"] if c["type"] == kind)
        candidates = sorted(SRC.rglob(cell["source"]))
        matches = []
        for p in candidates:
            raw = loadmat(p, simplify_cells=True)
            same = all(np.array_equal(raw[k], cell[k]) for k in ["R", "theta", "lambda"])
            obs = read_observations(p, m["lambdaTheta"])
            amplitude = np.sqrt(np.maximum(obs, 0))
            matches.append(
                {
                    "path": str(p.relative_to(ROOT)),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "embedded_cell_exact_match": same,
                    "max_amplitude_difference_from_saved_observations": float(
                        np.max(abs(amplitude - m["data"][:, col]))
                    ),
                    "negative_raw_values_on_fit_grid": int((obs < 0).sum()),
                }
            )
        assert any(
            c["embedded_cell_exact_match"]
            and c["max_amplitude_difference_from_saved_observations"] < 1e-14
            for c in matches
        )
        channels.append({"channel": kind, "candidates": matches})
    report["models"].append({"tag": tag, "channels": channels})
(OUT / "provenance-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
