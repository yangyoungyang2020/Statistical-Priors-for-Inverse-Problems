import pickle
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from configs.config import Config
from utils.plot_utils import PlotUtils
from forward_model.model import MethodOfMomentModel
from inverse_model.solve import InverseProblemSolver


def get_grid_permittivity(grid_positions):

    center_x = 0.4
    center_y = 0.4
    radius = 0.15
    epsilon_r = 1.1

    m = Config.doi["forward_grids"]
    scatterer = np.ones((m, m), dtype=float)

    # circle
    scatterer[(grid_positions[0] - center_x) ** 2 + (grid_positions[1] - center_y) ** 2 <= radius ** 2] = epsilon_r

    # square
    # mask = ((grid_positions[0] <= center_x + radius) & (grid_positions[0] >= center_x - radius) &
    #         (grid_positions[1] <= center_y + radius) & (grid_positions[1] >= center_y - radius))
    # scatterer[mask] = epsilon_r

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    original = ax1.imshow(np.real(scatterer), cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
    ax1.title.set_text("Scatterer: real")

    reconstructed_real = ax2.imshow(np.imag(scatterer), cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig.colorbar(reconstructed_real, ax=ax2, fraction=0.046, pad=0.04)
    ax2.title.set_text("Scatterer: imaginary")

    return scatterer


def solve_many(A, data, prior):
    values = np.linspace(0, 1e3, 20)
    fig1, axes1 = plt.subplots(nrows=4, ncols=5)
    fig2, axes2 = plt.subplots(nrows=4, ncols=5)

    real_rec = []
    imag_rec =  []
    for value in values:
        params = {"alpha": value}
        chi_real, chi_imag = InverseProblemSolver.solve(A, data, prior, params)
        real_rec.append(chi_real)
        imag_rec.append(chi_imag)

    fig1.subplots_adjust(hspace=0.2)
    fig1.suptitle('Reconstructions for real part')

    for ax, rec, in zip(axes1.flatten(), real_rec):
        ax.imshow(rec, cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())

    fig2.subplots_adjust(hspace=0.2)
    fig2.suptitle('Reconstructions for imaginary part')

    for ax, rec, in zip(axes2.flatten(), imag_rec):
        ax.imshow(rec, cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())


def plot_results(scatterer, chi_real, chi_imag):

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    original = ax1.imshow(scatterer, cmap=plt.cm.hot)
    fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
    ax1.title.set_text("Original scatterer")

    reconstructed_real = ax2.imshow(chi_real, cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig.colorbar(reconstructed_real, ax=ax2, fraction=0.046, pad=0.04)
    ax2.title.set_text("Real Reconstruction")

    reconstructed_imag = ax3.imshow(chi_imag, cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig.colorbar(reconstructed_imag, ax=ax3, fraction=0.046, pad=0.04)
    ax3.title.set_text("Imaginary Reconstruction")


if __name__ == '__main__':

    """" Solve the Forward Problem """
    # Instantiate forward model
    forward = MethodOfMomentModel()

    """" Generate / Read the scatterer """
    scatterer = get_grid_permittivity(forward.grid_positions)

    # Generate forward data for scatterer
    direct_field, direct_power, total_field, total_power = forward.generate_forward_data(scatterer)

    """" Solve the Inverse Problem """

    model_type = "prytov_complex"
    prior = "qs2D_complex"
    params = {"alpha": 10}

    A = InverseProblemSolver.get_inverse_model(model_type)
    data = InverseProblemSolver.get_measurement_data(model_type, direct_power, total_power)

    chi_real, chi_imag = InverseProblemSolver.solve(A, data, prior, params)
    plot_results(scatterer, chi_real, chi_imag)

    # solve_many(A, data, prior)
