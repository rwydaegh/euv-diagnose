import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

import euv_diagnose.optics as optics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/results"
base = optics.transfer_matrix


def explicit(
    thickness,
    nk,
    roughness,
    wavelength,
    angle,
    *,
    periods=0,
    caps=None,
    polarization="s",
    periodic_convention="continuous",
):
    if not periods:
        return base(thickness, nk, roughness, wavelength, angle, polarization=polarization)
    thickness = np.asarray(thickness)
    roughness = np.asarray(roughness)
    return base(
        np.r_[thickness[:caps], np.tile(thickness[caps:], periods)],
        np.column_stack([nk[:, :caps], np.tile(nk[:, caps:], (1, periods))]),
        np.r_[roughness[:caps], np.tile(roughness[caps:-1], periods), roughness[-1]],
        wavelength,
        angle,
        polarization=polarization,
    )


m = replace(
    optics.ReleasedModel.load(
        ROOT / "research/sources/sherwin/FIlm models/Reflectivityapp_workspace_fit131.mat"
    ),
    periodic_convention="continuous",
)
report = {}
for name in ["probe", "tight"]:
    d = np.load(OUT / f"phase-continuous-{name}.npz")
    report[name] = {}
    for kind in ["nominal", "candidate"]:
        with patch.object(optics, "transfer_matrix", explicit):
            r = m.amplitude(d[kind + "_x"])
        err = float(np.max(abs(r - d[kind + "_amplitude"])))
        if err > 1e-12:
            raise RuntimeError(f"{name}/{kind}: explicit-layer disagreement {err:g}")
        report[name][kind + "_max_complex_error"] = err
(OUT / "continuous-witness-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
