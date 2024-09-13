import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor

from euv_diagnose.learning import phase_error








def test_phase_errors_cross_branch_cut():
    np.testing.assert_allclose(
        phase_error([179.0, -179.0, 540.0], [-179.0, 179.0, 0.0]), [-2.0, 2.0, -180.0]
    )


