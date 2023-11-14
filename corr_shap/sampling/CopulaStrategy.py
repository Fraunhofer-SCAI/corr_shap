import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm

from .GaussStrategy import GaussStrategy


class CopulaStrategy(GaussStrategy):
    """ Uses a Gaussian copula to modify the kernel SHAP method to explain the output of any function.

    Using the Gaussian copula, we represent the data by their marginal distribution
    and transform them so that they are multivariate normally distributed.
    Now we can apply the first modification.
    After that, the data must be transformed back accordingly.
    """
    def __init__(self, explainer, **kwargs):
        """
        Construct all necessary attributes for the CopulaStrategy object.
        A sorted version of data is determined.
        """

        super().__init__(explainer)

        # sort each column of the data separately to use in transform_data_back
        self.data_sorted = np.zeros((self.data.shape[0], self.data.shape[1]))
        for col in range(self.data.shape[1]):
            sample = list(self.data[:, col])
            self.data_sorted[:, col] = sorted(sample)

    def sample(self, m, x=None, mean=None, cov=None):
        """
        Return prepared sample data.
        Samples from data that are previously transformed to normal distribution.
        Afterwards back transformation of data and samples.

        :param m: given mask of subset
        :param x: given instance to be explained
        :param mean: mean of given data
        :param cov: covariance of given data
        :return: samples with fixed masked features and normalized weights
        """

        samples, weights = super().sample(m, x, mean, cov)
        samples = self.transform_data_back(samples)
        return samples, weights

    def calc_mean_cov(self, data, weights):
        """
        Return mean and covariance of given data.

        :param data: data for which mean and covariance is to be calculated
        :param weights: weights assigned to data
        :return: mean and covariance
        """
        data_transformed = self.transform_data(data)
        return super().calc_mean_cov(data=data_transformed, weights=weights)

    def transform_data_back(self, data):
        """
        Transform normal distributed data used in copula method back into original distribution
        based on earlier margin distribution of features.
        """
        data = data.copy()
        unif_back = norm.cdf(data)
        unif_back[unif_back == 1] = 0.9999999999999999
        sort_index = (self.data_sorted.shape[0] * unif_back).astype(int)
        for col in range(self.data_sorted.shape[1]):
            data[:, col] = self.data_sorted[sort_index[:,col], col]
        return data

    def transform_data(self, data):
        """
        Transform given data into normal distributed data for copula method based on margin distribution of features.
        """
        # transform data and instance to uniform distributed data based on empirical distribution
        unif = np.zeros(data.shape)
        for col in range(data.shape[1]):
            self.ecdf = ECDF(data[:, col])
            unif[:, col] = self.ecdf(data[:, col])
        # to avoid errors: Set 1 to 0.999... and 0 to 0.0...01
        unif = np.where(unif != 1, unif, 0.9999999999999999)
        unif = np.where(unif != 0, unif, 0.0000000000000001)
        # transform uniform distributed data and instance to normal distributed data
        return norm.ppf(unif)

    def set_instance(self, instance):
        """ Set instance x to transformed version."""
        super().set_instance(instance)
        self.x = self.transform_instance(self.x)

    def transform_instance(self, x):
        """ Determine transformed version of instance x based on marginal distribution. """
        x_unif = np.zeros_like(x)
        for col in range(self.data.shape[1]):
            x_unif[:, col] = self.ecdf(x[:, col])
        x_unif = np.where(x_unif != 1, x_unif, 0.9999999999999999)
        x_unif = np.where(x_unif != 0, x_unif, 0.0000000000000001)
        return norm.ppf(x_unif)
