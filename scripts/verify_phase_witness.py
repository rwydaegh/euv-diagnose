"""Regenerate candidate checks with the original solver (requires GNU Octave)."""

import subprocess
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/results"
for name in ["probe", "tight"]:
    data = np.load(OUT / f"phase-ambiguity-{name}.npz")
    savemat(OUT / f"phase-ambiguity-{name}-input.mat", {"x": data["candidate_x"]}, oned_as="column")
subprocess.run(
    ["octave", "--no-gui", "--quiet", "scripts/verify_phase_witness.m"], cwd=ROOT, check=True
)
for name in ["probe", "tight"]:
    data = np.load(OUT / f"phase-ambiguity-{name}.npz")
    ref = loadmat(OUT / f"phase-ambiguity-{name}-reference.mat", simplify_cells=True)
    delta = float(np.max(abs(np.column_stack([ref["a"], ref["b"]]) - data["candidate_amplitude"])))
    if delta >= 1e-12:
        raise RuntimeError(f"{name}: original-code mismatch {delta:g}")
    print(f"{name}: original-code max complex-amplitude difference {delta:.3g}")
