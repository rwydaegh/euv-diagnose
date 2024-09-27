import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/results"
d = np.load(OUT / "phase-continuous-tight.npz")
summary = json.loads((OUT / "phase-continuous-tight.json").read_text())
grid = d["grid"]
sel = grid[:, 1] == 6
wl = grid[sel, 0]
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
for col, label in enumerate(["Absorber", "Multilayer"]):
    axes[0, col].plot(
        wl, abs(d["nominal_amplitude"][sel, col]) ** 2, color="#334155", lw=2, label="Nominal model"
    )
    axes[0, col].plot(
        wl,
        abs(d["candidate_amplitude"][sel, col]) ** 2,
        color="#54788a",
        ls="--",
        lw=1.8,
        label="Alternative model",
    )
    axes[0, col].set(
        title=label + " reflectivity at 6°", xlabel="Wavelength (nm)", ylabel="Reflected fraction"
    )
    axes[0, col].legend()
for col, label in enumerate(["Absorber", "Multilayer"]):
    diff = (
        abs(d["candidate_amplitude"][sel, col]) ** 2 - abs(d["nominal_amplitude"][sel, col]) ** 2
    ) / d["sigma"][col]
    axes[1, 0].plot(wl, diff, label=label, color=["#54788a", "#a18450"][col])
axes[1, 0].set(
    title="Intensity differences normalized by assumed noise",
    xlabel="Wavelength (nm)",
    ylabel="Intensity difference / assumed σ",
)
axes[1, 0].legend()
audit = json.loads((OUT / "phase-continuous-angle-audit.json").read_text())
for name, label, color in [
    ("probe", "Released bounds", "#54788a"),
    ("tight", "Concentrations within ±10%", "#a18450"),
]:
    witness = audit["witnesses"][name]
    axes[1, 1].plot(
        witness["angle_deg"], witness["pairwise_spectral_distance"], label=label, color=color
    )
axes[1, 1].axvspan(2, 8, color="#e2e8f0", alpha=0.7, label="Range used to construct pair")
axes[1, 1].set(
    title="Pairwise spectral distance by measurement angle",
    xlabel="Angle from normal (degrees)",
    ylabel="Total whitened distance for one angle block",
)
axes[1, 1].legend(fontsize=8)
fig.suptitle(
    f"Synthetic example: nearly matching spectra, {summary['phase_difference_deg']:.2f}° relative-phase difference\nAt 13.5 nm and 6°; illustrative noise/material bounds, not measured phase accuracy",
    fontsize=13,
)
fig.savefig(OUT / "phase-ambiguity.png", dpi=160)
print("Wrote phase-ambiguity.png from saved witness results")
