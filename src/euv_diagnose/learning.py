from pathlib import Path

import numpy as np


def phase_error(prediction, truth):
    return (np.asarray(prediction) - np.asarray(truth) + 180) % 360 - 180


def conformal_radius(errors, coverage=0.9):
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or not len(errors) or not np.isfinite(errors).all():
        raise ValueError("Errors must be a nonempty finite vector")
    if (errors < 0).any() or not 0 < coverage < 1:
        raise ValueError("Invalid errors or coverage")
    rank = int(np.ceil((len(errors) + 1) * coverage))
    if rank > len(errors):
        return float("inf")
    return float(np.partition(errors, rank - 1)[rank - 1])


class PhaseEnsemble:
    def __init__(self, arrays):
        self.arrays = arrays

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            return cls({key: archive[key].copy() for key in archive.files})

    def predict(self, spectra):
        data = self.arrays
        spectra = np.asarray(spectra, dtype=float)
        if spectra.ndim == 1:
            spectra = spectra[None, :]
        if (
            spectra.ndim != 2
            or spectra.shape[1] != len(data["input_mean"])
            or not np.isfinite(spectra).all()
        ):
            raise ValueError("Expected finite spectra on the model input grid")
        features = (spectra - data["input_mean"]) / data["input_scale"]
        predictions = []
        for member in range(int(data["members"])):
            hidden = features
            for layer in range(int(data["layers"])):
                hidden = hidden @ data[f"w_{member}_{layer}"] + data[f"b_{member}_{layer}"]
                if layer + 1 < int(data["layers"]):
                    hidden = np.maximum(hidden, 0)
            predictions.append(hidden[:, 0])
        return np.mean(predictions, axis=0) * data["target_scale"] + data["target_mean"]

    def interval(self, spectra, coverage=0.9):
        matches = np.flatnonzero(np.isclose(self.arrays["levels"], coverage))
        if len(matches) != 1:
            raise ValueError("Coverage level was not calibrated")
        prediction = self.predict(spectra)
        radius = float(self.arrays["radii"][matches[0]])
        return np.column_stack([prediction - radius, prediction + radius])
