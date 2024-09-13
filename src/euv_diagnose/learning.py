from pathlib import Path

import numpy as np


def phase_error(prediction, truth):
    return (np.asarray(prediction) - np.asarray(truth) + 180) % 360 - 180




