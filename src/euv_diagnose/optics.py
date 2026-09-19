from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def transfer_matrix(
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
    thickness = np.asarray(thickness, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    angle = np.asarray(angle, dtype=float)
    nk = np.asarray(nk, dtype=complex)
    roughness = np.asarray(roughness, dtype=float)
    if thickness.ndim != 1 or wavelength.ndim != 1 or len(wavelength) == 0:
        raise ValueError("Expected layer vector and nonempty wavelength vector")
    n, layers = len(wavelength), len(thickness)
    if nk.shape != (n, layers) or angle.shape != (n,) or roughness.shape != (layers + 1,):
        raise ValueError("Incompatible optical-array dimensions")
    if not all(np.isfinite(a).all() for a in (thickness, nk, roughness, wavelength, angle)):
        raise ValueError("Nonfinite optical input")
    if (
        (thickness < 0).any()
        or (roughness < 0).any()
        or (wavelength <= 0).any()
        or (np.abs(angle) >= 90).any()
    ):
        raise ValueError("Outside physical input domain")
    caps = layers if caps is None else caps
    if (
        not isinstance(periods, int | np.integer)
        or not isinstance(caps, int | np.integer)
        or periods < 0
        or not 0 <= caps <= layers
    ):
        raise ValueError("Invalid layer repetition")
    if periodic_convention not in ("continuous", "legacy"):
        raise ValueError("Unknown periodic convention")
    if periods == 0 and caps != layers:
        raise ValueError("A nonrepeated stack must propagate all supplied layers")
    if periods and caps == layers:
        raise ValueError("Repeated stack requires a nonempty period")
    if polarization not in ("s", "p"):
        raise ValueError("Only pure s or p polarization is supported")
    nk = np.column_stack([np.ones(n), nk, np.ones(n)])
    normal = np.sqrt(nk**2 - np.sin(np.deg2rad(angle))[:, None] ** 2)
    admittance = normal if polarization == "s" else nk**2 / normal
    kz = -2 * np.pi / wavelength[:, None] * normal
    p = (admittance[:, 1:] + admittance[:, :-1]) / (2 * admittance[:, :-1])
    m = -(admittance[:, 1:] - admittance[:, :-1]) / (2 * admittance[:, :-1])
    p *= np.exp(-((np.diff(kz, axis=1) * roughness) ** 2) / 2)
    m *= np.exp(-(((kz[:, 1:] + kz[:, :-1]) * roughness) ** 2) / 2)
    interfaces = np.empty((n, layers + 1, 2, 2), complex)
    interfaces[:, :, 0, 0] = interfaces[:, :, 1, 1] = p
    interfaces[:, :, 0, 1] = interfaces[:, :, 1, 0] = m
    phase = kz[:, 1:-1] * thickness
    propagation = np.zeros((n, layers, 2, 2), complex)
    propagation[:, :, 0, 0] = np.exp(-1j * phase)
    propagation[:, :, 1, 1] = np.exp(1j * phase)
    out = interfaces[:, 0].copy()
    for k in range(caps):
        out = out @ propagation[:, k] @ interfaces[:, k + 1]
    if periods and periodic_convention == "legacy":
        period = np.broadcast_to(np.eye(2, dtype=complex), (n, 2, 2)).copy()
        for k in range(caps, layers):
            period = period @ propagation[:, k] @ interfaces[:, k + 1]
        out = out @ np.linalg.matrix_power(period, periods)
    elif periods:
        body = np.broadcast_to(np.eye(2, dtype=complex), (n, 2, 2)).copy()
        for k in range(caps, layers):
            body = body @ propagation[:, k]
            if k + 1 < layers:
                body = body @ interfaces[:, k + 1]
        left = admittance[:, -2]
        right = admittance[:, caps + 1]
        kl = kz[:, -2]
        kr = kz[:, caps + 1]
        sigma = roughness[caps]
        bp = (right + left) / (2 * left) * np.exp(-(((kr - kl) * sigma) ** 2) / 2)
        bm = -(right - left) / (2 * left) * np.exp(-(((kr + kl) * sigma) ** 2) / 2)
        boundary = np.empty((n, 2, 2), complex)
        boundary[:, 0, 0] = boundary[:, 1, 1] = bp
        boundary[:, 0, 1] = boundary[:, 1, 0] = bm
        out = out @ np.linalg.matrix_power(body @ boundary, periods - 1) @ body @ interfaces[:, -1]
    if not np.isfinite(out).all():
        raise FloatingPointError("Transfer matrix overflow; requested stack not supported")
    return out


@dataclass
class ReleasedModel:
    x: np.ndarray
    indices: dict
    grid: np.ndarray
    factors: np.ndarray
    amu: np.ndarray
    density: np.ndarray
    periods: int
    caps: int
    per: int
    etch: int
    polarization: str
    observed_amplitude: np.ndarray
    periodic_convention: str = "legacy"

    @classmethod
    def load(cls, path: str | Path):
        m = loadmat(path, simplify_cells=True)["model"]
        return cls(
            m["x"].copy(),
            {k: np.asarray(v, dtype=int) - 1 for k, v in m["ind_struct"].items()},
            m["lambdaTheta"],
            m["f0f1_elements_r"],
            m["amu"],
            m["rho_nom"],
            int(m["N_ML"]),
            int(m["n_cap"]),
            int(m["n_per"]),
            int(m["n_etch"]),
            str(m["pol"]),
            m["data"],
        )

    def amplitude(self, x=None, *, angle_shift=0.0):
        x = self.x if x is None else np.asarray(x, dtype=float)
        if x.shape != self.x.shape or not np.isfinite(x).all():
            raise ValueError("Invalid parameter vector")
        thick = x[self.indices["thick"]].copy()
        rough = x[self.indices["rough"]]
        comp = x[self.indices["composition"]].reshape(len(self.amu), -1, order="F")
        if (comp < 0).any() or (rough < 0).any():
            raise ValueError("Negative material concentration or roughness")
        wl = self.grid[:, 0]
        angle = self.grid[:, 1] + angle_shift
        const = 1e-12 * wl**2 * 2.8179e-15 / (2 * np.pi)
        nk = 1 - const[:, None] * (
            self.factors @ ((self.density / self.amu * 6.0221409e23)[:, None] * comp)
        )
        etch_index = self.caps + self.per
        abs_indices = np.arange(self.etch + 1)
        shared = np.arange(self.etch + 1, etch_index)
        carbon = np.arange(etch_index + 1, len(thick))
        mirror_thick = np.r_[0.0, thick[shared]]
        mirror_nk = np.column_stack([np.ones(len(wl)), nk[:, shared]])
        mirror_rough = np.r_[0.0, rough[shared], 0.0]
        mirror = transfer_matrix(
            mirror_thick,
            mirror_nk,
            mirror_rough,
            wl,
            angle,
            periods=self.periods,
            caps=len(mirror_thick) - self.per,
            polarization=self.polarization,
            periodic_convention=self.periodic_convention,
        )
        corr = float(x[self.indices["substrate_roughness"]])
        if not 0 <= corr <= 1 / 0.2878:
            raise ValueError("Empirical correlated roughness correction outside domain")
        mirror[:, 1, 0] *= np.sqrt(1 - corr * 0.2878)
        absorber = transfer_matrix(
            thick[abs_indices],
            nk[:, abs_indices],
            np.r_[rough[abs_indices], 0.0],
            wl,
            angle,
            polarization=self.polarization,
        )

        remain = abs_indices[self.etch - 1 :]
        et = thick[remain].copy()
        et[0] = thick[etch_index]
        er = rough[remain].copy()
        er[0] = rough[etch_index]
        et = np.r_[thick[carbon], et]
        en = np.column_stack([nk[:, carbon], nk[:, remain]])
        er = np.r_[rough[carbon], er, 0.0]
        et = np.r_[sum(thick[abs_indices]) - sum(et), et]
        en = np.column_stack([np.ones(len(wl)), en])
        er = np.r_[0.0, er]
        negative = np.flatnonzero(et < 0)
        if len(negative):
            j = negative[0]
            if j + 1 >= len(et):
                raise ValueError("Invalid signed etch offset")
            et[j + 1] += et[j]
            et[j] = 0.0
        etched = transfer_matrix(et, en, er, wl, angle, polarization=self.polarization)
        matrices = [absorber @ mirror, etched @ mirror]
        result = np.column_stack([m[:, 1, 0] / m[:, 0, 0] for m in matrices])
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite reflection amplitude")
        return result

    def intensity(self, x=None, *, angle_shift=0.0):
        return np.abs(self.amplitude(x, angle_shift=angle_shift)) ** 2
