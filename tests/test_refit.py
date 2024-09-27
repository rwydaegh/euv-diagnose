from dataclasses import dataclass

import numpy as np
import pytest

from euv_diagnose.refit import StackBounds, fit_stack


@dataclass
class LinearOptics:
    x: np.ndarray
    grid: np.ndarray
    factors: np.ndarray
    observed_amplitude: np.ndarray
    periodic_convention: str = "continuous"

    def intensity(self, x, *, angle_shift=0):
        return x[0] * self.factors


def fixture():
    model = LinearOptics(
        np.array([1.0]),
        np.array([[13.0, 2], [13.1, 3], [13.2, 6]]),
        np.array([[0.1, 0.4], [0.2, 0.5], [0.15, 0.45]]),
        np.zeros((3, 2)),
    )
    return model, StackBounds(np.array([0.1]), np.array([2.0]), ["thickness"])


def test_holdout_does_not_change_fit_or_training_scales():
    model, bounds = fixture()
    observations = model.intensity([1.4])
    changed = observations.copy()
    changed[-1] = [-12, 500]
    mask = np.array([True, True, False])
    first = fit_stack(model, observations, mask, bounds, starts=1)
    second = fit_stack(model, changed, mask, bounds, starts=1)
    assert first["best"]["parameters"] == second["best"]["parameters"]
    assert first["channel_scale"] == second["channel_scale"]
    np.testing.assert_allclose(first["best"]["parameters"], [1.4], atol=1e-7)
    assert second["best"]["holdout_rmse"][0] > 12


def test_legacy_physics_is_rejected():
    model, bounds = fixture()
    model.periodic_convention = "legacy"
    with pytest.raises(ValueError, match="continuous-boundary"):
        fit_stack(model, model.intensity([1.4]), [True, True, False], bounds)


def test_invalid_bounds_and_empty_training_rejected():
    with pytest.raises(ValueError, match="bounds"):
        StackBounds([2], [1], ["x"])
    model, bounds = fixture()
    with pytest.raises(ValueError, match="Invalid fitting"):
        fit_stack(model, model.intensity([1.4]), [False] * 3, bounds)


def test_recorded_fit_reproduces_continuous_predictions():
    import json
    from pathlib import Path

    from euv_diagnose.optics import ReleasedModel

    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "research/results/refit-measured.json").read_text())
    model = ReleasedModel.load(
        root / "research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit131.mat"
    )
    model.periodic_convention = "continuous"
    for result in report["fits"]:
        best = result["best"]
        prediction = model.intensity(
            best["parameters"], angle_shift=best["angle_offset_deg"]
        ) * np.array(best["gains"])
        np.testing.assert_allclose(prediction, best["prediction_reflectivity"], atol=1e-12)
        train = model.grid[:, 1] <= 5
        observed = np.array(report["observed_reflectivity"])
        np.testing.assert_allclose(
            np.sqrt(np.mean((prediction[~train] - observed[~train]) ** 2, axis=0)),
            best["holdout_rmse"],
            atol=1e-12,
        )
