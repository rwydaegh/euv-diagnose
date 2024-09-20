"""Check portable inference and calibration mechanics independently of training."""

import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor

from euv_diagnose.learning import PhaseEnsemble, conformal_radius, phase_error


def test_portable_network_matches_sklearn(tmp_path):
    rng = np.random.default_rng(13)
    x = rng.normal(size=(80, 3))
    y = x[:, 0] * x[:, 1] + 0.2 * x[:, 2]
    estimator = MLPRegressor(
        hidden_layer_sizes=(8, 4), solver="lbfgs", max_iter=2000, random_state=8
    ).fit(x, y)
    arrays = {
        "input_mean": np.array([0.1, 0.2, -0.1]),
        "input_scale": np.array([2.0, 0.5, 1.0]),
        "target_mean": 3.0,
        "target_scale": 2.0,
        "members": 1,
        "layers": 3,
        "levels": np.array([0.9]),
        "radii": np.array([1.2]),
    }
    for layer, (weights, bias) in enumerate(
        zip(estimator.coefs_, estimator.intercepts_, strict=True)
    ):
        arrays[f"w_0_{layer}"] = weights
        arrays[f"b_0_{layer}"] = bias
    path = tmp_path / "model.npz"
    np.savez_compressed(path, **arrays)
    portable = PhaseEnsemble.load(path)
    observation = rng.normal(size=(7, 3))
    expected = (
        estimator.predict((observation - arrays["input_mean"]) / arrays["input_scale"]) * 2 + 3
    )
    np.testing.assert_allclose(portable.predict(observation), expected, rtol=1e-12, atol=1e-12)
    interval = portable.interval(observation)
    np.testing.assert_allclose(interval[:, 0], expected - 1.2)
    np.testing.assert_allclose(interval[:, 1], expected + 1.2)
    with pytest.raises(ValueError, match="Coverage"):
        portable.interval(observation, 0.8)
    with pytest.raises(ValueError, match="finite spectra"):
        portable.predict(np.ones((3, 2)))
    with pytest.raises(ValueError, match="finite spectra"):
        portable.predict(np.array([np.nan, 1.0, 2.0]))


def test_conformal_quantile_uses_finite_sample_rank():
    # ceil((9+1)*.9)=9; a naive empirical percentile would be too small.
    assert conformal_radius(np.arange(1.0, 10.0), 0.9) == 9
    assert conformal_radius(np.arange(1.0, 10.0), 0.5) == 5
    assert np.isinf(conformal_radius(np.arange(1.0, 10.0), 0.99))


@pytest.mark.parametrize("errors,level", [([], 0.9), ([np.nan], 0.9), ([-1.0], 0.9), ([1.0], 1.0)])
def test_invalid_calibration_rejected(errors, level):
    with pytest.raises(ValueError):
        conformal_radius(errors, level)


def test_phase_errors_cross_branch_cut():
    np.testing.assert_allclose(
        phase_error([179.0, -179.0, 540.0], [-179.0, 179.0, 0.0]), [-2.0, 2.0, -180.0]
    )


def test_published_checkpoint_reproduces_evaluation():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    model = PhaseEnsemble.load(root / "artifacts/model-phase-ensemble.npz")
    with np.load(root / "research/results/learning-evaluation.npz", allow_pickle=False) as data:
        for regime in ("nominal", "shifted"):
            np.testing.assert_allclose(
                model.predict(data[f"{regime}_spectra"]),
                data[f"{regime}_prediction"],
                rtol=1e-10,
                atol=1e-10,
            )
        errors = abs(
            phase_error(model.predict(data["calibration_spectra"]), data["calibration_phase"])
        )
        expected = [conformal_radius(errors, level) for level in model.arrays["levels"]]
        np.testing.assert_allclose(model.arrays["radii"], expected, rtol=1e-10, atol=1e-10)
