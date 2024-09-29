import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .optics import ReleasedModel

PARAMETERS = ("carbon_nm", "gain", "angle_offset_deg", "background")
LOWER = np.array([0.0, 0.8, -0.3, -0.005])
UPPER = np.array([6.0, 1.2, 0.3, 0.005])
SCALE = np.array([1.0, 0.1, 0.1, 0.001])


def predict(model: ReleasedModel, parameters):
    carbon, gain, angle, background = parameters
    x = model.x.copy()
    x[model.indices["thick"][-1]] = carbon
    return gain * model.intensity(x, angle_shift=angle)[:, 1] + background


@dataclass
class FitResult:
    parameters: np.ndarray
    prediction: np.ndarray
    seconds: float
    evaluations: int
    success: bool
    active_bounds: list[str]
    train_rmse: float


def fit(model, observed, train, *, mode="joint", starts=3, seed=42):
    observed = np.asarray(observed, dtype=float)
    train = np.asarray(train, dtype=bool)
    if (
        observed.shape != (len(model.grid),)
        or train.shape != observed.shape
        or not train.any()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("Invalid measurements or fitting mask")
    active = {"carbon": [0], "calibration": [1, 2, 3], "joint": [0, 1, 2, 3]}[mode]
    if starts < 1:
        raise ValueError("At least one start is required")
    fixed = np.array([model.x[model.indices["thick"][-1]], 1.0, 0.0, 0.0])
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    evaluations = 0

    def unpack(z):
        p = fixed.copy()
        p[active] = z
        return p

    def residual(z):
        nonlocal evaluations
        evaluations += 1
        return (predict(model, unpack(z)) - observed)[train]

    best = None
    for i in range(starts):
        start = fixed[active] if i == 0 else rng.uniform(LOWER[active], UPPER[active])
        result = least_squares(
            residual,
            start,
            bounds=(LOWER[active], UPPER[active]),
            x_scale=SCALE[active],
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=250,
        )
        if best is None or result.cost < best.cost:
            best = result
    p = unpack(best.x)
    prediction = predict(model, p)
    return FitResult(
        p,
        prediction,
        time.perf_counter() - t0,
        evaluations,
        bool(best.success),
        [PARAMETERS[k] for k, v in zip(active, best.active_mask, strict=True) if v],
        float(np.sqrt(np.mean((prediction[train] - observed[train]) ** 2))),
    )
