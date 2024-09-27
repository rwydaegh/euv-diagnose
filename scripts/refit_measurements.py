import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from euv_diagnose.data import read_observations
from euv_diagnose.optics import ReleasedModel
from euv_diagnose.refit import StackBounds, fit_stack

ROOT = Path(__file__).resolve().parents[1]


def git_metadata(root: Path) -> dict:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"code_revision": "source-archive", "working_tree_has_changes": None}
    return {"code_revision": revision, "working_tree_has_changes": dirty}


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the continuous-boundary, angle-split fit of measured mask 131."
    )
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--max-nfev", type=int, default=1200)
    args = parser.parse_args()
    source = ROOT / "research/sources/sherwin"
    model_path = source / "FIlm models/Reflectivityapp_workspace_fit131.mat"
    paths = [
        source / "Reflectivityapp/TaN_131/Data" / name
        for name in (
            "Reflectivityapp_workspace_2019_08_21_Abs.mat",
            "Reflectivityapp_workspace_2019_06_22_ML.mat",
        )
    ]
    model = ReleasedModel.load(model_path)
    model.periodic_convention = "continuous"
    table = loadmat(model_path, simplify_cells=True)["table"]
    names = [f"{i}:{row[0]}:{row[1]}:{row[2]}" for i, row in enumerate(table)]
    bounds = StackBounds(table[:, 3].astype(float), table[:, 6].astype(float), names)
    observed = np.column_stack([read_observations(path, model.grid) for path in paths])
    train = model.grid[:, 1] <= 5
    result = {
        "schema_version": 1,
        "configuration": {"starts": args.starts, "max_nfev": args.max_nfev, "seed": 131},
        "implementation_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                ROOT / "src/euv_diagnose/refit.py",
                ROOT / "src/euv_diagnose/optics.py",
                ROOT / "src/euv_diagnose/data.py",
            ]
        },
        **git_metadata(ROOT),
        "case": "measured-mask-131-initial",
        "split_id": "exploratory-angle-2-5-train-6-8-holdout-v1",
        "train_angles_deg": [2, 3, 4, 5],
        "holdout_angles_deg": [6, 7, 8],
        "channels": ["absorber", "multilayer"],
        "wavelength_nm": model.grid[:, 0].tolist(),
        "angle_deg": model.grid[:, 1].tolist(),
        "observed_reflectivity": observed.tolist(),
        "negative_observations_preserved": int((observed < 0).sum()),
        "parameter_names": names,
        "lower_bounds": bounds.lower.tolist(),
        "upper_bounds": bounds.upper.tolist(),
        "sources": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in [model_path, *paths]
        ],
        "assumptions": [
            "Continuous-boundary solver; released optical factors, pure s polarization and empirical roughness correction retained.",
            "All 22 coordinates free in the upstream table are varied within its bounds; other coordinates fixed.",
            "Residuals are divided by each channel's training-only RMS. This is an objective, not calibrated experimental noise.",
            "Saved fit seeds the optimization and previously saw held-out angles. Evaluation is exploratory, not blind.",
            "Calibration sensitivity assumes separate channel gains 0.95–1.05 and a shared angle offset ±0.1 degrees; these are not instrument specifications.",
            "Nominal-angle phase predictions have no independent experimental ground truth. Multistart spread is not uncertainty coverage.",
        ],
        "fits": [],
    }
    out = ROOT / "research/results/refit-measured.json"
    for calibration in [False, True]:
        fit = fit_stack(
            model,
            observed,
            train,
            bounds,
            calibration=calibration,
            starts=args.starts,
            max_nfev=args.max_nfev,
        )
        result["fits"].append(fit)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(
            fit["method"],
            {
                key: fit["best"][key]
                for key in [
                    "cost",
                    "success",
                    "train_rmse",
                    "holdout_rmse",
                    "phase_deg_13_5nm_6deg",
                ]
            },
            flush=True,
        )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for channel, title in enumerate(["Absorber", "Multilayer"]):
        selected = model.grid[:, 1] == 6
        wavelength = model.grid[selected, 0]
        axes[0, channel].plot(
            wavelength,
            observed[selected, channel],
            "o",
            ms=3,
            color="#334155",
            label="Measured, withheld 6°",
        )
        for fit, color in zip(result["fits"], ["#087f8c", "#ce6a35"], strict=True):
            prediction = np.array(fit["best"]["prediction_reflectivity"])[selected, channel]
            axes[0, channel].plot(
                wavelength, prediction, color=color, label=fit["method"].replace("_", " ")
            )
            axes[1, channel].plot(wavelength, prediction - observed[selected, channel], color=color)
        axes[0, channel].set_title(title)
        axes[0, channel].set_ylabel("Reflectivity")
        axes[1, channel].set_ylabel("Prediction − observation")
        axes[1, channel].set_xlabel("Wavelength (nm)")
        axes[1, channel].axhline(0, color="#64748b", lw=0.7)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Real EUV measurements · continuous-boundary refit\nTrain angles 2–5°; exploratory holdout 6–8°"
    )
    figure.tight_layout()
    figure.savefig(ROOT / "research/results/refit-measured.png", dpi=170)


if __name__ == "__main__":
    main()
