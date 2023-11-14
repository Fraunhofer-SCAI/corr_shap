from .SamplingStrategy import SamplingStrategy
from .GaussStrategy import GaussStrategy
from .CopulaStrategy import CopulaStrategy
from .EmpiricalStrategy import EmpiricalStrategy
from .GaussEmpiricalStrategy import GaussEmpiricalStrategy
from .CopulaEmpiricalStrategy import CopulaEmpiricalStrategy


def get_sampling_strategy(type, explainer,  kwargs):
    """Assign the sampling strategy method to the explainer based on the given type. """
    sampling_strategies = {"default": SamplingStrategy, "gauss": GaussStrategy, "copula": CopulaStrategy,
                           "empirical": EmpiricalStrategy, "gauss+empirical": GaussEmpiricalStrategy,
                           "copula+empirical": CopulaEmpiricalStrategy}
    return sampling_strategies[type](explainer=explainer, **kwargs)
