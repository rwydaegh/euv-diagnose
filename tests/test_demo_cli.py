import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from euv_diagnose.cli import main
from euv_diagnose.demo import build_bundle, rectangular_spectra
from euv_diagnose.optics import ReleasedModel

ROOT = Path(__file__).resolve().parents[1]


def test_rectangular_export_preserves_signed_observations_and_order():
    grid = np.array([[13.5, 6], [13.4, 2], [13.4, 6], [13.5, 2]])
    values = np.array([[0.1, 0.6], [-0.001, 0.5], [0.2, 0.4], [0.3, 0.7]])
    wavelengths, angles, result = rectangular_spectra(grid, values)
    assert wavelengths == [13.4, 13.5]
    assert angles == [2, 6]
    assert result[0][0][0] == -0.001
    assert result[1][1] == [0.1, 0.6]


@pytest.mark.parametrize(
    "grid",
    [np.array([[13.5, 6], [13.5, 6]]), np.array([[13.5, 6], [13.4, 2]])],
)
def test_export_rejects_ambiguous_or_incomplete_grids(grid):
    with pytest.raises(ValueError):
        rectangular_spectra(grid, np.ones((2, 2)))


def test_export_matches_saved_scientific_result(tmp_path):
    path = tmp_path / "data.json"
    data = build_bundle(ROOT, path)
    loaded = json.loads(path.read_text())
    assert loaded == data
    case = loaded["synthetic"]
    assert len(case["angles"]) == 29
    assert len(case["wavelengths"]) == 101
    nominal = np.array(case["nominal"]["spectra"])
    alternative = np.array(case["alternative"]["spectra"])
    sigma = np.array(case["sigma"])
    distance = np.linalg.norm((alternative[:7] - nominal[:7]) / sigma)
    assert distance == pytest.approx(case["baselineDistance"], rel=1e-10)
    assert 0.99 < case["phaseDifferenceDeg"] < 1.01
    model = replace(
        ReleasedModel.load(
            ROOT / "research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit131.mat"
        ),
        periodic_convention="continuous",
    )
    index = np.flatnonzero(np.isclose(model.grid[:, 0], 13.5) & (model.grid[:, 1] == 6))[0]
    with np.load(ROOT / "research/results/phase-continuous-tight.npz", allow_pickle=False) as saved:
        for label, key in [("nominal", "nominal_x"), ("alternative", "candidate_x")]:
            amplitude = model.amplitude(saved[key])[index]
            calculated = np.angle(amplitude[0] / amplitude[1], deg=True)
            difference = (calculated - case[label]["phaseDeg"] + 180) % 360 - 180
            assert difference == pytest.approx(0, abs=1e-10)
    assert len(loaded["measured"]["comparisons"]) == 2
    assert len(loaded["learning"]["evaluations"][0]["truth"]) == 500


def test_cli_reports_missing_checkout_without_traceback(tmp_path, capsys):
    assert main(["--workspace", str(tmp_path), "inspect"]) == 2
    assert "checkout" in capsys.readouterr().err


def test_cli_export(tmp_path, capsys):
    output = tmp_path / "demo.json"
    assert main(["--workspace", str(ROOT), "export", "--output", str(output)]) == 0
    assert output.is_file()
    assert "Wrote" in capsys.readouterr().out
