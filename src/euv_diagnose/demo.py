"""Build the browser's small, deterministic result bundle from scientific artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from .optics import ReleasedModel


def revision(root: Path) -> str:
    """Return an informative revision without requiring Git in a release archive."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "source-archive"


def rectangular_spectra(grid: np.ndarray, values: np.ndarray) -> tuple[list, list, list]:
    """Convert unordered wavelength/angle observations to the browser's dense grid."""
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 2 or values.shape != (len(grid), 2):
        raise ValueError("Expected N wavelength/angle pairs and N two-channel intensities")
    if not np.isfinite(grid).all() or not np.isfinite(values).all():
        raise ValueError("Cannot export nonfinite spectra")
    wavelengths, angles = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    spectra = np.full((len(angles), len(wavelengths), 2), np.nan)
    seen = set()
    for point, value in zip(grid, values, strict=True):
        location = (
            int(np.searchsorted(angles, point[1])),
            int(np.searchsorted(wavelengths, point[0])),
        )
        if location in seen:
            raise ValueError("Duplicate wavelength/angle pair")
        seen.add(location)
        spectra[location] = value
    if not np.isfinite(spectra).all():
        raise ValueError("Browser bundle requires a complete wavelength/angle grid")
    return wavelengths.tolist(), angles.tolist(), spectra.tolist()


def synthetic_bundle(root: Path) -> dict:
    """Recompute continuous-boundary spectra from the saved synthetic witness."""
    result_path = root / "research/results/phase-continuous-tight.npz"
    report_path = root / "research/results/phase-continuous-tight.json"
    report = json.loads(report_path.read_text())
    if report["periodic_convention"] != "continuous":
        raise ValueError("The public demo requires continuous periodic boundaries")
    with np.load(result_path, allow_pickle=False) as data:
        nominal_x, candidate_x = data["nominal_x"], data["candidate_x"]
        sigma = data["sigma"]
    model = replace(
        ReleasedModel.load(
            root / "research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit131.mat"
        ),
        periodic_convention="continuous",
    )
    wavelengths = np.unique(model.grid[:, 0])
    angles = np.arange(2, 31)
    nominal, alternative, distances = [], [], []
    for angle in angles:
        grid = model.grid[: len(wavelengths)].copy()
        grid[:, 1] = angle
        current = replace(model, grid=grid, factors=model.factors[: len(wavelengths)])
        a = current.intensity(nominal_x)
        b = current.intensity(candidate_x)
        nominal.append(a.tolist())
        alternative.append(b.tolist())
        distances.append(float(np.linalg.norm((b - a) / sigma)))
    return {
        "wavelengths": wavelengths.tolist(),
        "angles": angles.tolist(),
        "nominal": {"spectra": nominal, "phaseDeg": report["nominal_phase_deg"]},
        "alternative": {
            "spectra": alternative,
            "phaseDeg": report["nominal_phase_deg"] + report["phase_difference_deg"],
        },
        "sigma": sigma.tolist(),
        "phaseDifferenceDeg": report["phase_difference_deg"],
        "baselineDistance": report["total_whitened_spectral_distance"],
        "distances": distances,
        "metadata": {
            "title": "Two spectra. Two possible phases.",
            "description": "Synthetic models constructed from the released TaN mask family.",
            "sourceUrl": "https://github.com/s-sherwin/EUV",
            "artifactSha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "assumptions": [
                "Synthetic intensities; the phase difference is known by construction.",
                "Material concentrations vary within ±10% of nominal; thicknesses use the released bounds.",
                "Illustrative independent intensity noise: σ = 0.0001 absorber, 0.001 multilayer.",
                "The initial comparison uses all spectra at 2–8°; both models use continuous layer boundaries.",
                "Extra-angle separation compares this pair only. It is not a posterior probability or a globally optimal experiment.",
            ],
        },
    }


def measured_bundle(root: Path) -> dict:
    """Expose both fitted assumptions with unchanged signed measured intensities."""
    path = root / "research/results/refit-measured.json"
    report = json.loads(path.read_text())
    grid = np.column_stack([report["wavelength_nm"], report["angle_deg"]])
    wavelengths, angles, observed = rectangular_spectra(grid, report["observed_reflectivity"])
    comparisons = []
    for fit in report["fits"]:
        best = fit["best"]
        _, _, predicted = rectangular_spectra(grid, best["prediction_reflectivity"])
        calibrated = fit["method"] == "stack_and_calibration"
        comparisons.append(
            {
                "id": fit["method"],
                "label": "Stack + calibration" if calibrated else "Stack only",
                "predicted": predicted,
                "phaseDeg": best["phase_deg_13_5nm_6deg"],
                "rmse": best["holdout_rmse"],
                "detail": (
                    "Two gains within ±5% and a shared angle offset within ±0.1°. Sensitivity assumptions, not instrument specifications."
                    if calibrated
                    else "22 released free stack parameters; calibration held fixed."
                ),
                "converged": best["success"],
                "activeBounds": best["active_bounds"],
            }
        )
    first = comparisons[0]
    return {
        "label": "Measured TaN mask · continuous-boundary refit",
        "wavelengths": wavelengths,
        "angles": angles,
        "observed": observed,
        "predicted": first["predicted"],
        "comparisons": comparisons,
        "trainingAngles": report["train_angles_deg"],
        "heldoutAngles": report["holdout_angles_deg"],
        "rmse": first["rmse"],
        "metadata": {
            "sourceUrl": "https://github.com/s-sherwin/EUV",
            "artifactSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "description": "Initial measured absorber and multilayer spectra, refitted with the corrected periodic boundaries. Compare phase predictions under two calibration assumptions.",
            "limitations": report["assumptions"],
        },
    }


def learning_bundle(root: Path) -> dict:
    """Export executed evaluation metrics and predictions, never training claims alone."""
    path = root / "research/results/learning-benchmark.json"
    report = json.loads(path.read_text())
    nominal = report["metrics"]["nominal"]
    shifted = report["metrics"]["shifted"]
    interval = next(x for x in nominal["neural"]["intervals"] if x["nominal"] == 0.9)
    shifted_interval = next(x for x in shifted["neural"]["intervals"] if x["nominal"] == 0.9)
    count = report["splits"]["nominal"]["count"]
    physical = report["physical_subset_comparison"]
    physical_seconds = float(np.median([x["seconds"] for x in report["physical_baseline"]]))
    with np.load(root / "research/results/learning-evaluation.npz", allow_pickle=False) as data:
        evaluations = [
            {
                "name": "Nominal" if split == "nominal" else "Shifted",
                "truth": data[split + "_phase"].tolist(),
                "prediction": data[split + "_prediction"].tolist(),
                "intervalHalfWidthDeg": interval["width_deg"] / 2,
            }
            for split in ["nominal", "shifted"]
        ]
    return {
        "title": "A trained model, with its limits measured",
        "description": "Five neural networks estimate phase from synthetic spectra on a fixed grid. Independent calibration sets interval widths; a changed-material and correlated-error challenge tests their limits.",
        "metrics": [
            {
                "label": "Neural phase RMSE",
                "value": f"{nominal['neural']['rmse_deg']:.2f}°",
                "detail": f"{count} held-out nominal simulations",
            },
            {
                "label": "Ridge baseline RMSE",
                "value": f"{nominal['ridge']['rmse_deg']:.2f}°",
                "detail": "Same train and test data",
            },
            {
                "label": "90% interval coverage",
                "value": f"{100 * interval['coverage']:.1f}%",
                "detail": f"Nominal test; width {interval['width_deg']:.2f}°",
            },
            {
                "label": "Shifted coverage",
                "value": f"{100 * shifted_interval['coverage']:.1f}%",
                "detail": "The nominal interval guarantee does not survive this shift",
            },
            {
                "label": "Prediction latency",
                "value": f"{1000 * report['single_case_seconds']['neural']:.2f} ms",
                "detail": "One fixed-grid case on this CPU; excludes training",
            },
            {
                "label": "Training simulations",
                "value": f"{report['splits']['train']['count']:,}",
                "detail": f"{report['generation_seconds']:.1f}s generation + {report['training_seconds']:.1f}s neural training",
            },
            {
                "label": "Physical / neural subset RMSE",
                "value": f"{physical['physical_rmse_deg']:.2f}° / {physical['neural_rmse_deg']:.2f}°",
                "detail": f"Same {physical['count']} test cases; physical fits are more accurate",
            },
            {
                "label": "Physical fit median time",
                "value": f"{physical_seconds:.2f} s",
                "detail": "Three starts per case; speed–accuracy tradeoff",
            },
        ],
        "evaluations": evaluations,
        "limitations": [
            "Synthetic phase labels come from the continuous-boundary model. Experimental phase accuracy has not been established.",
            "Intervals describe nominal marginal coverage under the calibration distribution, not a posterior over layer structures.",
            "Shifted material assumptions and correlated errors reduce coverage; the model has no validated out-of-distribution detector.",
            "The fixed input grid contains 63 wavelength/angle pairs and both channels. Missing measurements are not supported.",
            "Classical physical fits are evaluated separately on a smaller subset; a ridge comparison alone does not establish superiority over physical inference.",
        ],
        "sourceUrl": "https://github.com/s-sherwin/EUV",
        "artifactSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_bundle(root: Path, output: Path) -> dict:
    """Write a self-contained browser bundle from available completed studies."""
    bundle = {"schemaVersion": 1, "revision": revision(root), "synthetic": synthetic_bundle(root)}
    if (root / "research/results/refit-measured.json").is_file():
        bundle["measured"] = measured_bundle(root)
    if (root / "research/results/learning-benchmark.json").is_file():
        bundle["learning"] = learning_bundle(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, allow_nan=False, separators=(",", ":")) + "\n")
    return bundle
