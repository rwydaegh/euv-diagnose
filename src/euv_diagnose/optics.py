from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def transfer_matrix(thickness, nk, roughness, wavelength, angle, *, periods=0, caps=None, polarization='s', periodic_convention='continuous'):
    thickness = np.asarray(thickness, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    angle = np.asarray(angle, dtype=float)
    nk = np.asarray(nk, dtype=complex)
    roughness = np.asarray(roughness, dtype=float)
    (n, layers) = (len(wavelength), len(thickness))
    if nk.shape != (n, layers) or angle.shape != (n,) or roughness.shape != (layers + 1,):
        raise ValueError('Incompatible optical-array dimensions')
    if not all((np.isfinite(a).all() for a in (thickness, nk, roughness, wavelength, angle))):
        raise ValueError('Nonfinite optical input')
    if (thickness < 0).any() or (roughness < 0).any() or (wavelength <= 0).any() or (np.abs(angle) >= 90).any():
        raise ValueError('Outside physical input domain')
    caps = layers if caps is None else caps
    if periodic_convention not in ('continuous', 'legacy'):
        raise ValueError('Unknown periodic convention')
    if periods and caps == layers:
        raise ValueError('Repeated stack requires a nonempty period')
    if polarization not in ('s', 'p'):
        raise ValueError('Only pure s or p polarization is supported')
    nk = np.column_stack([np.ones(n), nk, np.ones(n)])
    normal = np.sqrt(nk ** 2 - np.sin(np.deg2rad(angle))[:, None] ** 2)
    admittance = normal if polarization == 's' else nk ** 2 / normal
    kz = -2 * np.pi / wavelength[:, None] * normal
    p = (admittance[:, 1:] + admittance[:, :-1]) / (2 * admittance[:, :-1])
    m = -(admittance[:, 1:] - admittance[:, :-1]) / (2 * admittance[:, :-1])
    p *= np.exp(-(np.diff(kz, axis=1) * roughness) ** 2 / 2)
    m *= np.exp(-((kz[:, 1:] + kz[:, :-1]) * roughness) ** 2 / 2)
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
    if periods:
        period = np.broadcast_to(np.eye(2, dtype=complex), (n, 2, 2)).copy()
        for k in range(caps, layers):
            period = period @ propagation[:, k] @ interfaces[:, k + 1]
        out = out @ np.linalg.matrix_power(period, periods)
    if not np.isfinite(out).all():
        raise FloatingPointError('Transfer matrix overflow; requested stack not supported')
    return out
