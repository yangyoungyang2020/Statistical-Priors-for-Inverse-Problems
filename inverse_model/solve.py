from inverse_model.inverse import LinearInverse
from inverse_model.models import PRytov, PRytovComplex
from priors.tikhonov import TikhonovPriors
from priors.sparsity import SparsityPriors


class InverseProblemSolver:

    @staticmethod
    def get_model_class(model_name):
        if model_name == "prytov":
            model_class = PRytov()
        elif model_name == "prytov_complex":
            model_class = PRytovComplex()
        else:
            raise ValueError("Incorrect model name, input should be either 'prytov' or 'prytov_complex'")
        return model_class

    @staticmethod
    def get_inverse_model(model_name):
        inverse_problem = LinearInverse()
        direct_field = inverse_problem.get_direct_field()
        incident_field = inverse_problem.get_incident_field()
        integral_values = inverse_problem.get_greens_integral()

        model_class = InverseProblemSolver.get_model_class(model_name)
        A = model_class.get_model(direct_field, incident_field, integral_values)
        return A

    @staticmethod
    def get_measurement_data(model_name, direct_power, total_power):
        model_class = InverseProblemSolver.get_model_class(model_name)
        y = model_class.get_data(total_power, direct_power)
        return y

    @staticmethod
    def get_regularizer(prior):
        mapping = {
            "lasso": SparsityPriors.lasso,
            "elastic_net": SparsityPriors.elastic_net,
            "ridge": TikhonovPriors.ridge,
            "ridge_complex": TikhonovPriors.ridge_complex,
            "qs2D": TikhonovPriors.quadratic_smoothing_2d,
            "qs2D_complex": TikhonovPriors.quadratic_smoothing_2d_complex
        }
        if prior not in mapping.keys():
            raise ValueError("Invalid prior")
        return mapping[prior]

    @staticmethod
    def solve(model, data, prior, params):
        regularizer = InverseProblemSolver.get_regularizer(prior)
        chi = regularizer(model, data, params)
        return chi
