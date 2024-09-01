"""Construct a bounded, exact-model phase ambiguity witness on synthetic data.

The nominal model comes from a real release; observations here are noiseless
simulations. Noise scales are illustrative assumptions, not experimental estimates.
This finds a candidate pair, not a posterior, confidence interval, or global limit.
"""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares

from euv_diagnose.optics import ReleasedModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--material-fraction",
    type=float,
    default=None,
    help="Optionally restrict each free concentration to this fractional range around nominal; an illustrative sensitivity assumption.",
)
parser.add_argument("--target-deg", type=float, default=1.0)
parser.add_argument("--output-name", default="phase-ambiguity-probe")
parser.add_argument("--periodic-convention", choices=["legacy", "continuous"], default="legacy")
args = parser.parse_args()
if args.material_fraction is not None and not 0 < args.material_fraction < 1:
    parser.error("material-fraction must be between zero and one")
if not np.isfinite(args.target_deg):
    parser.error("target-deg must be finite")
if Path(args.output_name).name != args.output_name:
    parser.error("output-name must be a filename stem")
ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit131.mat"
m = replace(ReleasedModel.load(path), periodic_convention=args.periodic_convention)
table = loadmat(path, simplify_cells=True)["table"]
lo = table[:, 3].astype(float)
hi = table[:, 6].astype(float)
if args.material_fraction is not None:
    for k in np.flatnonzero((hi > lo) & (table[:, 1] == "Concentration")):
        lo[k] = max(lo[k], m.x[k] * (1 - args.material_fraction))
        hi[k] = min(hi[k], m.x[k] * (1 + args.material_fraction))
ids = np.flatnonzero(hi > lo)
scale = (hi - lo)[ids]
z0 = (m.x[ids] - lo[ids]) / scale
r0 = m.amplitude()
y0 = abs(r0) ** 2
idx = np.flatnonzero(np.isclose(m.grid[:, 0], 13.5) & (m.grid[:, 1] == 6))[0]
q0 = r0[idx, 0] / r0[idx, 1]
sigma = np.array([1e-4, 1e-3])


def evaluate(z):
    x = m.x.copy()
    x[ids] = lo[ids] + scale * z
    r = m.amplitude(x)
    residual = ((abs(r) ** 2 - y0) / sigma).ravel()
    phase = np.angle((r[idx, 0] / r[idx, 1]) / q0)
    return residual, phase, x, r


# Local calculation is an exploratory diagnostic only; constraints can invalidate
# extrapolation of its unbounded covariance, hence the exact nonlinear check below.
h = 1e-5
J = []
g = []
for k in range(len(ids)):
    zp = z0.copy()
    zm = z0.copy()
    zp[k] += h
    zm[k] -= h
    yp, qp, _, _ = evaluate(zp)
    ym, qm, _, _ = evaluate(zm)
    J.append((yp - ym) / (2 * h))
    g.append(np.angle(np.exp(1j * (qp - qm))) / (2 * h))
J = np.array(J).T
g = np.array(g)
local = {}
for name, cols in [
    ("carbon_only", [0, 9]),
    ("thicknesses", list(range(10))),
    ("all_released_free", list(range(len(ids)))),
]:
    A = J[:, cols]
    u, s, v = np.linalg.svd(A, full_matrices=False)
    local[name] = {
        "scaled_jacobian_condition": float(s[0] / s[-1]),
        "linearized_unbounded_phase_sd_deg": float(np.rad2deg(np.linalg.norm((v @ g[cols]) / s))),
    }
# A modest requested phase separation, not an optimized headline number.
target_deg = args.target_deg
target = np.deg2rad(target_deg)
phase_weight = 1e4
calls = 0


def residual(z):
    global calls
    calls += 1
    e, phi, _, _ = evaluate(z)
    return np.r_[e, phase_weight * (phi - target)]


start = time.perf_counter()
fit = least_squares(
    residual,
    z0,
    bounds=(np.zeros(len(ids)), np.ones(len(ids))),
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-9,
    max_nfev=350,
)
e, phi, x, r = evaluate(fit.x)
report = {
    "scope": "Synthetic exact-model witness search, fixed calibration. Released free-parameter bounds are a permissive feasibility box, not a validated material prior. No global ambiguity bound, posterior, or experimental accuracy claim.",
    "periodic_convention": args.periodic_convention,
    "material_fraction_restriction": args.material_fraction,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "nominal_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "assumed_independent_intensity_sigma": sigma.tolist(),
    "free_parameter_indices_zero_based": ids.tolist(),
    "parameter_labels": [" / ".join(str(s) for s in table[k, :3]) for k in ids],
    "lower": lo[ids].tolist(),
    "upper": hi[ids].tolist(),
    "nominal_parameters": m.x[ids].tolist(),
    "candidate_parameters": x[ids].tolist(),
    "nominal_phase_deg": float(np.angle(q0, deg=True)),
    "phase_target_deg": target_deg,
    "phase_difference_deg": float(np.rad2deg(phi)),
    "total_whitened_spectral_distance": float(np.linalg.norm(e)),
    "per_channel_max_abs_intensity_difference": np.max(abs(abs(r) ** 2 - y0), axis=0).tolist(),
    "per_channel_intensity_rmse": np.sqrt(np.mean((abs(r) ** 2 - y0) ** 2, axis=0)).tolist(),
    "linear_diagnostic": local,
    "optimizer_success": bool(fit.success),
    "optimizer_message": fit.message,
    "active_bounds": [int(ids[k]) for k in np.flatnonzero(fit.active_mask)],
    "forward_calls_optimization": calls,
    "seconds_optimization": time.perf_counter() - start,
}
out = ROOT / "research/results"
(out / (args.output_name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
np.savez_compressed(
    out / (args.output_name + ".npz"),
    grid=m.grid,
    nominal_amplitude=r0,
    candidate_amplitude=r,
    nominal_x=m.x,
    candidate_x=x,
    sigma=sigma,
)
print(json.dumps(report, indent=2))
