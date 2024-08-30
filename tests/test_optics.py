from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from euv_diagnose.optics import ReleasedModel, transfer_matrix

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("tag", ["131", "073", "074"])
def test_original_complex_amplitude(tag):
    m = ReleasedModel.load(
        ROOT / f"research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit{tag}.mat"
    )
    d = loadmat(ROOT / f"research/results/sherwin-replay-{tag}.mat", simplify_cells=True)
    np.testing.assert_allclose(
        m.amplitude(), np.column_stack([d["r_abs"], d["r_etch"]]), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("tag", ["131", "073", "074"])
@pytest.mark.parametrize("pol", ["s", "p"])
def test_perturbed_original_complex_amplitude(tag, pol):
    m = ReleasedModel.load(
        ROOT / f"research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit{tag}.mat"
    )
    m = replace(m, polarization=pol)
    x = m.x.copy()
    x[m.indices["thick"][0]] += 0.03
    x[m.indices["thick"][-1]] += 0.07
    x[m.indices["rough"]] += 0.02
    d = loadmat(ROOT / f"research/results/optics-reference-{tag}-{pol}.mat", simplify_cells=True)
    np.testing.assert_allclose(
        m.amplitude(x, angle_shift=0.03), np.column_stack([d["a"], d["b"]]), rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("pol", ["s", "p"])
def test_lossless_quarter_wave_slab(pol):
    # Analytic normal-incidence quarter-wave film n=1.5 between vacuum.
    n = 1.5
    wl = np.array([600.0])
    mat = transfer_matrix(
        [600 / (4 * n)], np.array([[n]]), [0, 0], wl, np.array([0.0]), polarization=pol
    )
    r = mat[:, 1, 0] / mat[:, 0, 0]
    t = 1 / mat[:, 0, 0]
    expected_R = ((n * n - 1) / (n * n + 1)) ** 2
    np.testing.assert_allclose(abs(r) ** 2, expected_R, atol=1e-14)
    np.testing.assert_allclose(abs(r) ** 2 + abs(t) ** 2, 1, atol=1e-14)


def test_vacuum_and_zero_thickness_limits():
    for thick, n in [(31, 1), (0, 1.7)]:
        mat = transfer_matrix([thick], np.array([[n]]), [0, 0], np.array([13.5]), np.array([17.0]))
        np.testing.assert_allclose(mat[:, 1, 0] / mat[:, 0, 0], 0, atol=1e-14)
        np.testing.assert_allclose(abs(1 / mat[:, 0, 0]) ** 2, 1, atol=1e-14)


def test_domain_rejection():
    with pytest.raises(ValueError):
        transfer_matrix([-1], np.ones((1, 1)), [0, 0], np.array([13.5]), np.array([0.0]))
    m = ReleasedModel.load(
        ROOT / "research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit131.mat"
    )
    with pytest.raises(ValueError):
        m.intensity(angle_shift=90)


@pytest.mark.parametrize("periods", [1, 2, 5])
@pytest.mark.parametrize("pol", ["s", "p"])
def test_repeated_cell_matches_explicit_layers(periods, pol):
    wl = np.array([500.0, 600.0])
    angles = np.array([13.0, 0.0])
    nk = np.array([[1.5 - 0.01j, 2.0 - 0.02j], [1.5, 2.0]])
    t = np.array([100.0, 75.0])
    rough = np.array([0.1, 0.2, 0.3])
    a = transfer_matrix(t, nk, rough, wl, angles, periods=periods, caps=0, polarization=pol)
    # First interface of each cell has rough[0], only the last exit has rough[-1].
    b = transfer_matrix(
        np.tile(t, periods),
        np.tile(nk, (1, periods)),
        np.r_[np.tile(rough[:-1], periods), rough[-1]],
        wl,
        angles,
        polarization=pol,
    )
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    expected = ((1.5 / 2.0) ** (2 * periods) - 1) ** 2 / ((1.5 / 2.0) ** (2 * periods) + 1) ** 2
    # Analytic quarter-wave stack, smooth interfaces.
    c = transfer_matrix(
        t, nk[1:2], np.zeros(3), wl[1:2], angles[1:2], periods=periods, caps=0, polarization=pol
    )
    np.testing.assert_allclose(abs(c[:, 1, 0] / c[:, 0, 0]) ** 2, expected, atol=1e-12)


def test_capped_periodic_stack_matches_explicit_layers():
    wl = np.array([13.5, 14.0])
    angles = np.array([6.0, 12.0])
    nk = np.array([[0.99 - 0.001j, 0.98 - 0.01j, 0.97 - 0.02j]] * 2)
    t = np.array([0.4, 3.5, 2.5])
    rough = np.array([0.01, 0.02, 0.03, 0.04])
    periods = 4
    a = transfer_matrix(t, nk, rough, wl, angles, periods=periods, caps=1)
    b = transfer_matrix(
        np.r_[t[:1], np.tile(t[1:], periods)],
        np.column_stack([nk[:, :1], np.tile(nk[:, 1:], (1, periods))]),
        np.r_[rough[:1], np.tile(rough[1:-1], periods), rough[-1]],
        wl,
        angles,
    )
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"caps": 0.5},
        {"periods": 1.5},
        {"periods": -1},
        {"caps": 0},
        {"periods": 1, "caps": 1},
        {"polarization": "mixed"},
        {"periodic_convention": "unknown"},
    ],
)
def test_unsupported_kernel_configuration_is_explicit(kwargs):
    with pytest.raises(ValueError):
        transfer_matrix(
            [1.0], np.array([[1.5]]), [0.0, 0.0], np.array([13.5]), np.array([0.0]), **kwargs
        )


def test_scalar_and_empty_wavelength_grids_are_rejected():
    for wl in [np.array(13.5), np.array([])]:
        with pytest.raises(ValueError):
            transfer_matrix([1.0], np.array([[1.5]]), [0.0, 0.0], wl, np.array([0.0]))
