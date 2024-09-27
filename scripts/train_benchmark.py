import argparse
import hashlib
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from euv_diagnose.learning import PhaseEnsemble, conformal_radius, phase_error
from euv_diagnose.optics import ReleasedModel

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit131.mat"


def setup():
    model = replace(ReleasedModel.load(SOURCE), periodic_convention="continuous")
    table = loadmat(SOURCE, simplify_cells=True)["table"]
    lower, upper = table[:, 3].astype(float), table[:, 6].astype(float)
    ids = np.flatnonzero(upper > lower)
    material = np.array([table[k, 1] == "Concentration" for k in ids])
    rows = np.flatnonzero(
        np.isin(model.grid[:, 1], [2, 5, 8])
        & np.isclose((model.grid[:, 0] - 12.5) / 0.1, np.round((model.grid[:, 0] - 12.5) / 0.1))
    )
    reference = np.flatnonzero(np.isclose(model.grid[:, 0], 13.5) & (model.grid[:, 1] == 6))[0]
    rows = np.r_[rows, reference]
    model = replace(model, grid=model.grid[rows], factors=model.factors[rows])
    amplitude = model.amplitude()
    phase_zero = float(np.angle(amplitude[-1, 0] / amplitude[-1, 1], deg=True))
    return model, ids, material, lower[ids], upper[ids], phase_zero


def bounds(model, ids, material, lower, upper, fraction=0.03):
    low, high = lower.copy(), upper.copy()
    low[material] = np.maximum(low[material], model.x[ids[material]] * (1 - fraction))
    high[material] = np.minimum(high[material], model.x[ids[material]] * (1 + fraction))
    return low, high


def simulate(model, ids, low, high, phase_zero, count, seed, shifted=False):
    rng = np.random.default_rng(seed)
    spectra, phases, parameters = [], [], []
    for _ in range(count):
        physical = rng.uniform(low, high)
        nuisance = np.r_[rng.uniform(0.98, 1.02, 2), rng.uniform(-0.1, 0.1)]
        x = model.x.copy()
        x[ids] = physical
        amplitude = model.amplitude(x, angle_shift=nuisance[2])
        values = abs(amplitude[:-1]) ** 2 * nuisance[:2]
        values += rng.normal(size=values.shape) * np.array([1e-4, 1e-3])
        if shifted:
            ripple = np.sin((model.grid[:-1, 0] - 12.5) * np.pi * 2 + rng.uniform(0, 2 * np.pi))
            values += ripple[:, None] * np.array([3e-4, 3e-3])

        target_model = replace(model, grid=model.grid[-1:], factors=model.factors[-1:])
        target = target_model.amplitude(x)[0]
        phase = np.angle(target[0] / target[1] * np.exp(-1j * np.deg2rad(phase_zero)), deg=True)
        spectra.append(values.ravel())
        phases.append(phase)
        parameters.append(np.r_[physical, nuisance])
    return np.asarray(spectra), np.asarray(phases), np.asarray(parameters)


def score(truth, prediction, radii, levels):
    error = phase_error(prediction, truth)
    result = {
        "mae_deg": float(np.mean(abs(error))),
        "rmse_deg": float(np.sqrt(np.mean(error**2))),
        "p95_absolute_error_deg": float(np.quantile(abs(error), 0.95)),
    }
    result["intervals"] = []
    for radius, level in zip(radii, levels, strict=True):
        hits = int(np.sum(abs(error) <= radius))
        n = len(error)
        p = hits / n
        z = 1.96
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        result["intervals"].append(
            {
                "nominal": float(level),
                "coverage": p,
                "coverage_wilson95": [center - half, center + half],
                "width_deg": float(2 * radius),
            }
        )
    return result


def plot_benchmark(evaluation_path, report_path, output_path):
    report = json.loads(Path(report_path).read_text())
    with np.load(evaluation_path, allow_pickle=False) as evaluation:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
        for ax, name, title in zip(
            axes, ("nominal", "shifted"), ("Nominal regime", "Shifted challenge"), strict=True
        ):
            actual = evaluation[f"{name}_phase"]
            predicted = evaluation[f"{name}_prediction"]
            ax.scatter(actual, predicted, s=10, alpha=0.5, color="#54788a")
            limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
            ax.plot(limits, limits, color="#777777", lw=1, ls="--")
            intervals = report["metrics"][name]["neural"]["intervals"]
            coverage = next(entry["coverage"] for entry in intervals if entry["nominal"] == 0.9)
            ax.set(
                title=f"{title}\nEmpirical coverage of nominal 90% intervals: {coverage:.1%}",
                xlabel="True phase offset (degrees)",
                ylabel="Neural estimate (degrees)",
            )
            ax.set_title(ax.get_title(), fontsize=10)
            ax.grid(alpha=0.15)
        fig.savefig(output_path)
        plt.close(fig)


def run(args):
    start = time.perf_counter()
    model, ids, material, lo, hi, phase_zero = setup()
    low, high = bounds(model, ids, material, lo, hi)
    shifted_low, shifted_high = bounds(model, ids, material, lo, hi, 0.08)
    splits = {}
    for name, count, seed in [
        ("train", args.train, 4101),
        ("calibration", 500, 4102),
        ("nominal", 500, 4103),
        ("shifted", 500, 4104),
    ]:
        shifted = name == "shifted"
        splits[name] = simulate(
            model,
            ids,
            shifted_low if shifted else low,
            shifted_high if shifted else high,
            phase_zero,
            count,
            seed,
            shifted,
        )
        print(f"Generated {name}: {count}", flush=True)
    generation_seconds = time.perf_counter() - start
    features, truth, _ = splits["train"]
    scaler = StandardScaler().fit(features)
    target_mean, target_scale = np.mean(truth), np.std(truth)
    train_features = scaler.transform(features)
    arrays = {
        "input_mean": scaler.mean_,
        "input_scale": scaler.scale_,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "members": args.members,
        "layers": 3,
        "grid": model.grid[:-1],
        "phase_reference_deg": phase_zero,
        "levels": np.array([0.5, 0.8, 0.9, 0.95]),
    }
    training_start = time.perf_counter()
    training_records = []
    for member in range(args.members):
        estimator = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            alpha=0.1,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            learning_rate_init=0.001,
            batch_size=128,
            random_state=900 + member,
        )
        estimator.fit(train_features, (truth - target_mean) / target_scale)
        for layer, (weight, bias) in enumerate(
            zip(estimator.coefs_, estimator.intercepts_, strict=True)
        ):
            arrays[f"w_{member}_{layer}"] = weight
            arrays[f"b_{member}_{layer}"] = bias
        training_records.append(
            {
                "iterations": estimator.n_iter_,
                "internal_validation_r2": estimator.best_validation_score_,
            }
        )
        print(f"Trained member {member}: {training_records[-1]}", flush=True)
    training_seconds = time.perf_counter() - training_start
    neural = PhaseEnsemble(arrays)
    levels = arrays["levels"]
    calibration_features, calibration_truth, _ = splits["calibration"]
    arrays["radii"] = np.array(
        [
            conformal_radius(
                abs(phase_error(neural.predict(calibration_features), calibration_truth)), p
            )
            for p in levels
        ]
    )
    baseline_start = time.perf_counter()
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 17), cv=5).fit(train_features, truth)
    ridge_seconds = time.perf_counter() - baseline_start
    ridge_radii = [
        conformal_radius(
            abs(
                phase_error(
                    ridge.predict(scaler.transform(calibration_features)), calibration_truth
                )
            ),
            p,
        )
        for p in levels
    ]
    metrics = {}
    for name in ("nominal", "shifted"):
        x, y, _ = splits[name]
        metrics[name] = {
            "neural": score(y, neural.predict(x), arrays["radii"], levels),
            "ridge": score(y, ridge.predict(scaler.transform(x)), ridge_radii, levels),
        }
    latency = {}
    sample = splits["nominal"][0][:1]
    for name, predict in [
        ("neural", neural.predict),
        ("ridge", lambda x: ridge.predict(scaler.transform(x))),
    ]:
        predict(sample)
        tick = time.perf_counter()
        for _ in range(500):
            predict(sample)
        latency[name] = (time.perf_counter() - tick) / 500
    physical_results = []
    physical_start = time.perf_counter()

    selection_rng = np.random.default_rng(715)
    subset = selection_rng.choice(500, args.physical_cases, replace=False)
    for index in subset:
        observed = splits["nominal"][0][index].reshape(-1, 2)
        sigma = np.array([1e-4, 1e-3])

        def residual(z, observed=observed, sigma=sigma):
            x = model.x.copy()
            x[ids] = low + (high - low) * z[: len(ids)]
            gain = 0.98 + 0.04 * z[len(ids) : len(ids) + 2]
            angle = -0.1 + 0.2 * z[-1]
            return ((model.intensity(x, angle_shift=angle)[:-1] * gain - observed) / sigma).ravel()

        tick = time.perf_counter()
        start_results = []
        for trial in range(3):
            initial = (
                np.full(len(ids) + 3, 0.5)
                if trial == 0
                else selection_rng.uniform(0.05, 0.95, len(ids) + 3)
            )
            candidate = least_squares(
                residual, initial, bounds=(0, 1), max_nfev=200, ftol=1e-7, xtol=1e-7, gtol=1e-7
            )
            start_results.append(candidate)
        fit = min(start_results, key=lambda result: result.cost)
        x = model.x.copy()
        x[ids] = low + (high - low) * fit.x[: len(ids)]
        target = replace(model, grid=model.grid[-1:], factors=model.factors[-1:]).amplitude(x)[0]
        phase = np.angle(target[0] / target[1] * np.exp(-1j * np.deg2rad(phase_zero)), deg=True)
        physical_results.append(
            {
                "index": int(index),
                "starts": 3,
                "start_success": [bool(result.success) for result in start_results],
                "total_nfev": sum(result.nfev for result in start_results),
                "seconds": time.perf_counter() - tick,
                "success": bool(fit.success),
                "nfev": fit.nfev,
                "prediction_deg": float(phase),
                "truth_deg": float(splits["nominal"][1][index]),
            }
        )
        print(f"Physical fit {index}: {physical_results[-1]}", flush=True)
    report = {
        "scope": "Synthetic phase-offset regression; full fixed measurement grid; no posterior or experimental accuracy claim.",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "threads": 1,
        "optics_sha256": hashlib.sha256(
            (ROOT / "src/euv_diagnose/optics.py").read_bytes()
        ).hexdigest(),
        "target_range_deg": {k: [float(v[1].min()), float(v[1].max())] for k, v in splits.items()},
        "physical_subset_seed": 715,
        "solver": "continuous",
        "grid": model.grid[:-1].tolist(),
        "phase_reference_deg": phase_zero,
        "target": "relative phase at 13.5 nm, 6 degrees; offset from nominal model",
        "parameter_indices": ids.tolist(),
        "lower": low.tolist(),
        "upper": high.tolist(),
        "shifted_lower": shifted_low.tolist(),
        "shifted_upper": shifted_high.tolist(),
        "noise_sigma": [1e-4, 1e-3],
        "gain_bounds": [0.98, 1.02],
        "angle_bounds_deg": [-0.1, 0.1],
        "splits": {
            k: {"count": len(v[1]), "seed": 4101 + i} for i, (k, v) in enumerate(splits.items())
        },
        "training": training_records,
        "generation_seconds": generation_seconds,
        "training_seconds": training_seconds,
        "ridge_training_seconds": ridge_seconds,
        "single_case_seconds": latency,
        "ridge_alpha": float(ridge.alpha_),
        "metrics": metrics,
        "physical_baseline": physical_results,
        "physical_baseline_total_seconds": time.perf_counter() - physical_start,
    }
    if physical_results:
        actual = np.array([p["truth_deg"] for p in physical_results])
        predicted = np.array([p["prediction_deg"] for p in physical_results])
        report["physical_subset_comparison"] = {
            "count": len(actual),
            "physical_rmse_deg": float(np.sqrt(np.mean(phase_error(predicted, actual) ** 2))),
            "neural_rmse_deg": float(
                np.sqrt(
                    np.mean(phase_error(neural.predict(splits["nominal"][0][subset]), actual) ** 2)
                )
            ),
            "ridge_rmse_deg": float(
                np.sqrt(
                    np.mean(
                        phase_error(
                            ridge.predict(scaler.transform(splits["nominal"][0][subset])), actual
                        )
                        ** 2
                    )
                )
            ),
        }
    out = ROOT / "research/results"
    out.mkdir(exist_ok=True)
    np.savez_compressed(ROOT / "artifacts/model-phase-ensemble.npz", **arrays)
    np.savez_compressed(
        out / "learning-evaluation.npz",
        **{
            f"{name}_{kind}": values[i]
            for name, values in splits.items()
            if name != "train"
            for i, kind in enumerate(("spectra", "phase", "parameters"))
        },
        nominal_prediction=neural.predict(splits["nominal"][0]),
        shifted_prediction=neural.predict(splits["shifted"][0]),
    )
    (out / "learning-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_benchmark(
        out / "learning-evaluation.npz",
        out / "learning-benchmark.json",
        out / "learning-benchmark.svg",
    )
    print(
        json.dumps(
            {
                "metrics": metrics,
                "timing": latency,
                "subset": report.get("physical_subset_comparison"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a fixed-grid synthetic EUV phase regressor on CPU."
    )
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--physical-cases", type=int, default=32)
    args = parser.parse_args()
    if args.train < 100 or args.members < 1 or args.physical_cases < 0 or args.physical_cases > 500:
        parser.error("Use at least 100 training cases, one member, and 0–500 physical cases")
    with threadpool_limits(limits=1):
        run(args)
