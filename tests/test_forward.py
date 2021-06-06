import os
import pytest
import numpy as np
from scipy.io import loadmat

from forward_model.model import MethodOfMomentModel


class TestForwardModel:

    test_data_path = os.path.join(os.path.dirname(__file__), "data")

    @staticmethod
    @pytest.mark.parametrize("index", [1, 2, 3, 4])
    def test_total_power(index):

        scatterer_data_path = os.path.join(TestForwardModel.test_data_path, f"scatterer{index}")

        scatterer = loadmat(os.path.join(scatterer_data_path, "scatterer.mat"))
        scatterer = scatterer["epsono_r"]
        total_power_mat = loadmat(os.path.join(scatterer_data_path, "total_power.mat"))
        total_power_mat = total_power_mat["Ptotal"]

        forward = MethodOfMomentModel()
        direct_field, direct_power, total_field, total_power = forward.generate_forward_data(scatterer)

        power_diff = total_power - total_power_mat

        assert np.max(power_diff) < 1e-10
