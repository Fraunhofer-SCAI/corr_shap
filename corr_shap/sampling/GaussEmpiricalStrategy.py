import numpy as np

from .GaussStrategy import GaussStrategy
from .EmpiricalStrategy import EmpiricalStrategy


class GaussEmpiricalStrategy(EmpiricalStrategy, GaussStrategy):
    """ Uses a combination of the Gauss method and the Emp-Cond method
    to modify the kernel SHAP method to explain the output of a function.

    Experiments showed that for small subset sizes the Emp-Cond method performs better
    and for larger subsets the Gauss or Copula method performs better.
    """

    def __init__(self, explainer, sigma=0.1, eta=0.9, dim=3, **kwargs):
        """
        Construct all necessary attributes for the CombiGaussStrategy object,
        especially the number of dimension that is used to determine which sampling method to use.
        """
        EmpiricalStrategy.__init__(self, explainer, sigma=sigma, eta=eta)
        # dimension that decides which method is used
        self.dim = dim

    def sample(self, m):
        """
        Determine correct sample method.
        If subset size (given by m) is smaller than a fixed dim sample with empirical conditional sample
        otherwise with gauss.

        :param m: given mask of subset
        :return: right sample strategy
        """
        if np.sum(m) <= self.dim:
            return EmpiricalStrategy.sample(self, m)
        else:
            return GaussStrategy.sample(self, m)
