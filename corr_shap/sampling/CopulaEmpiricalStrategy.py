import numpy as np

from .CopulaStrategy import CopulaStrategy
from .EmpiricalStrategy import EmpiricalStrategy

class CopulaEmpiricalStrategy(EmpiricalStrategy, CopulaStrategy):
    """ Uses a combination of the Copula method and the Emp-Cond method
        to modify the kernel SHAP method to explain the output of a function.

        Experiments showed that for small subset sizes the Emp-Cond method performs better
        and for larger subsets the Gauss or Copula method performs better.
        """

    def __init__(self, explainer, sigma=0.1, eta=0.9, dim=3, **kwargs):
        """
         Construct all necessary attributes for the CombiGaussStrategy object,
         especially the number of dimension that is used to determine which sampling method to use
         and the transformed version of mean and covariance.
         """
        EmpiricalStrategy.__init__(self, explainer, sigma=sigma, eta=eta)
        self.mean_transformed, self.cov_transformed = CopulaStrategy.calc_mean_cov(self, data=self.data, weights=self.data_weights)
        # dimension that decides which method is used
        self.dim = dim
    
    def calc_mean_cov(self, data, weights):
        """
        Return mean and covariance of given data.

        :param data: data for which mean and covariance is to be calculated
        :param weights: weights assigned to data
        :return: mean and covariance
        """
        return EmpiricalStrategy.calc_mean_cov(self, data, weights)

    def sample(self, m):
        """
        Determine correct sample method.
        If subset size (given by m) is smaller than an fixed dim sample with empirical conditional sample
        otherwise with copula.

        :param m: given mask of subset
        :return: right sample strategy
        """
        if np.sum(m) <= self.dim:
            return EmpiricalStrategy.sample(self, m)
        else:
            return CopulaStrategy.sample(self, m, x=self.x_transformed, mean=self.mean_transformed_varying, cov=self.cov_transformed_varying)

    def set_instance(self, instance):
        """ Set instance, transformed x and the transformed mean and covariance version."""
        EmpiricalStrategy.set_instance(self, instance)
        self.x_transformed = CopulaStrategy.transform_instance(self, self.x)
        self.mean_transformed_varying, self.cov_transformed_varying = self.mask_mean_cov(self.mean_transformed, self.cov_transformed)
