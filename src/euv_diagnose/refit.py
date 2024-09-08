"""Bounded experimental-stack refits with an explicit training-angle split.

The objective balances channel RMS, not a calibrated experimental likelihood.
Multistart dispersion is an optimization diagnostic, never a credible interval.
"""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from .optics import ReleasedModel


@dataclass
class StackBounds:
    """Bounds in the released 92-coordinate parameterization."""

    lower: np.ndarray
    upper: np.ndarray
    names: list[str]

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if (
            self.lower.ndim != 1
            or self.upper.shape != self.lower.shape
            or len(self.names) != len(self.lower)
            or not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or (self.lower > self.upper).any()
        ):
            raise ValueError("Invalid stack parameter bounds")

    @property
    def active(self):
        return np.flatnonzero(self.upper > self.lower)


def subset_model(model, mask):
    """Keep cached optical factors aligned when restricting the measured grid."""
    return replace(
        model,
        grid=model.grid[mask],
        factors=model.factors[mask],
        observed_amplitude=model.observed_amplitude[mask],
    )


def fit_stack(
    model: ReleasedModel,
    observed,
    train,
    bounds: StackBounds,
    *,
    calibration=False,
    starts=3,
    seed=131,
    max_nfev=250,
    initial=None,
):
    """Fit free stack coordinates, optionally channel gains and a shared angle.

    Calibration bounds (gain 0.95–1.05, angle ±0.1°) are sensitivity assumptions,
    not instrument specifications. Channel scales use training observations only.
    The first start is supplied/saved; others perturb it by 15% of bound widths.
    Fixed coordinates use the released table's fixed values.
    """
    observed = np.asarray(observed, dtype=float)
    train = np.asarray(train, dtype=bool)
    if (
        observed.shape != (len(model.grid), 2)
        or train.shape != (len(model.grid),)
        or not train.any()
        or not np.isfinite(observed).all()
        or bounds.lower.shape != model.x.shape
        or starts < 1
        or max_nfev < 1
    ):
        raise ValueError("Invalid fitting data, mask, bounds, or evaluation budget")
    if model.periodic_convention != "continuous":
        raise ValueError("Experimental refit requires continuous-boundary physics")
    active = bounds.active
    lower, width = bounds.lower[active], (bounds.upper - bounds.lower)[active]
    initial = model.x if initial is None else np.asarray(initial, dtype=float)
    if initial.shape != model.x.shape or not np.isfinite(initial).all():
        raise ValueError("Invalid initial stack")
    center = np.clip((initial[active] - lower) / width, 1e-6, 1 - 1e-6)
    if calibration:
        center = np.r_[center, 0.5, 0.5, 0.5]
    names = [bounds.names[i] for i in active]
    if calibration:
        names += ["absorber_gain", "multilayer_gain", "angle_offset_deg"]
    fit_model = subset_model(model, train)
    scale = np.sqrt(np.mean(observed[train] ** 2, axis=0))
    if (scale <= 0).any():
        raise ValueError("Both training channels need nonzero RMS")
    count = 0

    def unpack(z):
        x = bounds.lower.copy()
        x[active] = lower + width * z[: len(active)]
        gains = 0.95 + 0.1 * z[-3:-1] if calibration else np.ones(2)
        angle = -0.1 + 0.2 * z[-1] if calibration else 0.0
        return x, gains, float(angle)

    def residual(z):
        nonlocal count
        count += 1
        x, gains, angle = unpack(z)
        return (
            (fit_model.intensity(x, angle_shift=angle) * gains - observed[train]) / scale
        ).ravel()

    rng = np.random.default_rng(seed)
    runs = []
    started = perf_counter()
    for index in range(starts):
        z0 = (
            center
            if index == 0
            else np.clip(center + rng.normal(0, 0.15, len(center)), 1e-6, 1 - 1e-6)
        )
        before = count
        tick = perf_counter()
        solution = least_squares(
            residual,
            z0,
            bounds=(0, 1),
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=max_nfev,
        )
        x, gains, angle = unpack(solution.x)
        prediction = model.intensity(x, angle_shift=angle) * gains
        phase_index = np.flatnonzero(
            np.isclose(model.grid[:, 0], 13.5) & np.isclose(model.grid[:, 1], 6)
        )
        phase = None
        if len(phase_index):
            amplitudes = model.amplitude(x)[phase_index[0]]
            phase = float(np.rad2deg(np.angle(amplitudes[0] / amplitudes[1])))
        runs.append(
            {
                "start": index,
                "cost": float(solution.cost),
                "success": bool(solution.success),
                "status": int(solution.status),
                "message": solution.message,
                "nfev": solution.nfev,
                "forward_evaluations": count - before,
                "seconds": perf_counter() - tick,
                "optimality": float(solution.optimality),
                "active_bounds": [
                    names[i] for i, z in enumerate(solution.x) if z < 1e-4 or z > 1 - 1e-4
                ],
                "parameters": x.tolist(),
                "gains": gains.tolist(),
                "angle_offset_deg": angle,
                "phase_deg_13_5nm_6deg": phase,
                "prediction_reflectivity": prediction.tolist(),
                "residual_reflectivity": (prediction - observed).tolist(),
                "per_angle_rmse": {
                    str(float(angle_value)): np.sqrt(
                        np.mean(
                            (
                                prediction[model.grid[:, 1] == angle_value]
                                - observed[model.grid[:, 1] == angle_value]
                            )
                            ** 2,
                            axis=0,
                        )
                    ).tolist()
                    for angle_value in np.unique(model.grid[:, 1])
                },
                "train_rmse": np.sqrt(
                    np.mean((prediction[train] - observed[train]) ** 2, axis=0)
                ).tolist(),
                "holdout_rmse": np.sqrt(
                    np.mean((prediction[~train] - observed[~train]) ** 2, axis=0)
                ).tolist()
                if (~train).any()
                else None,
            }
        )
    best = min(runs, key=lambda result: result["cost"])
    return {
        "method": "stack_and_calibration" if calibration else "stack_only",
        "active_stack_parameters": active.tolist(),
        "active_parameter_names": names,
        "channel_scale": scale.tolist(),
        "seed": seed,
        "seconds": perf_counter() - started,
        "best_start": best["start"],
        "best": best,
        "starts": runs,
    }
