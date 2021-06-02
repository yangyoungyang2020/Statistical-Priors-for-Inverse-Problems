import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from configs.config import Config
from utils.plot_utils import PlotUtils
from forward_model.model import MethodOfMomentModel
from inverse_model.solve import InverseProblemSolver


def get_grid_permittivity(grid_positions):
    center_x = 0.3
    center_y = 0.3
    radius = 0.15
    epsilon_r = 2

    m = Config.doi["forward_grids"]
    scatterer = np.ones((m, m), dtype=float)

    # circle
    scatterer[(grid_positions[0] - center_x) ** 2 + (grid_positions[1] - center_y) ** 2 <= radius ** 2] = epsilon_r

    # square
    # mask = ((grid_positions[0] <= center_x + radius) & (grid_positions[0] >= center_x - radius) &
    #         (grid_positions[1] <= center_y + radius) & (grid_positions[1] >= center_y - radius))
    # scatterer[mask] = epsilon_r

    return scatterer


def plot_results(scatterer, chi_real, chi_imag):

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    original = ax1.imshow(scatterer, cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
    ax1.title.set_text("Original scatterer")

    reconstructed_real = ax2.imshow(chi_real, cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig.colorbar(reconstructed_real, ax=ax2, fraction=0.046, pad=0.04)
    ax2.title.set_text("Real Reconstruction")

    reconstructed_imag = ax3.imshow(chi_imag, cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig.colorbar(reconstructed_imag, ax=ax3, fraction=0.046, pad=0.04)
    ax3.title.set_text("Imaginary Reconstruction")


if __name__ == '__main__':

    # Forward problem

    # Instantiate forward model
    forward = MethodOfMomentModel()

    # Create scatterer
    scatterer = get_grid_permittivity(forward.grid_positions)

    # Generate forward data for scatterer
    direct_field, direct_power, total_field, total_power = forward.generate_forward_data(scatterer)

    # Inverse problem

    A = InverseProblemSolver.get_inverse_model()
    data = InverseProblemSolver.get_measurement_data(direct_power, total_power)

    prior = "ridge_complex"
    params = {"alpha": 1}
    chi_real, chi_imag = InverseProblemSolver.solve(A, data, prior, params)

    plot_results(scatterer, chi_real, chi_imag)
