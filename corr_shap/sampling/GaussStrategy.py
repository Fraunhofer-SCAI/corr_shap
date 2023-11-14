import numpy as np
import warnings

from .SamplingStrategy import SamplingStrategy


class GaussStrategy(SamplingStrategy):
    """Uses the idea that the data to be explained are multivariate normally distributed
    to modify the kernel SHAP method to explain the output of a function.

    If the data are multivariate normally distributed, we can calculate
    the conditional expected value and the conditional covariance while fixing the given subset.
    Sampling can be done using these new parameters.
    """

    def __init__(self, explainer):
        """ Construct all necessary attributes for the GaussStrategy object, especially mean and covariance."""
        super().__init__(explainer)
        self.mean_gen, self.cov_gen = self.calc_mean_cov(self.data, self.data_weights)

    def sample(self, m, x=None, mean=None, cov=None):
        """
        Return prepared sample data.
        Samples from normal distribution with computed conditional mean (_varying) und covariance (_varying)

        :param m: given mask of subset
        :param x: given instance to be explained
        :param mean: mean of given data
        :param cov: covariance of given data
        :return: samples with fixed masked features and normalized weights
        """

        if x is None:
            x = self.x
        if mean is None:
            mean = self.mean_varying
        if cov is None:
            cov = self.cov_varying

        mean_cond, cov_cond = self.calc_conditional_mean_cov(x, m, mean, cov)
        samples = self.data.copy()
        samples = self.set_masked_features_to_instance(m, x, samples)
        gauss_sample_len_mean = np.random.multivariate_normal(mean_cond, cov_cond, self.N)
        # if features are not in subset fill them with random gauss samples of same mean_varying and covariance
        mask = m == 1.0
        help_mask = np.zeros_like(self.group_mask)
        help_mask[self.group_mask == True] = ~mask
        samples[:, help_mask] = gauss_sample_len_mean

        weights = np.ones(self.N) / self.N
        return samples, weights

    def set_varying_feature_groups(self, varying_groups):
        """
        Set mask only to features which vary in the dataset.

        :param varying_groups: contain indices of featues which vary in dataset
        :return: set mask to true if feature varies, determines mean and covariance only from varying features
        """
        super().set_varying_feature_groups(varying_groups)
        group_mask = np.zeros(self.data.data.shape[1], dtype=bool)
        for ind in varying_groups:
            group_mask[ind] = True
        self.group_mask = group_mask
        self.mean_varying, self.cov_varying = self.mask_mean_cov(self.mean_gen, self.cov_gen)

    def calc_mean_cov(self, data, weights=None):
        """
        Return mean and covariance of given data.

        :param data: data for which mean and covariance is to be calculated
        :param weights: weights assigned to data
        :return: mean and covariance
        """
        mean_trainingset = np.average(data, axis=0, weights=weights)
        cov_trainingset = np.cov(np.transpose(data))

        return mean_trainingset, cov_trainingset

    def calc_conditional_mean_cov(self, x, m, mean, cov):
        """
        Return mean und covariance of non-subset features conditional on (with instance x) fixed features of subset.

        :param x: given instance to be explained
        :param m: given mask of subset
        :param mean: mean of given data
        :param cov: covariance of given data
        :return: conditional mean and covariance of non-subset features
        """
        #
        warnings.filterwarnings("ignore")

        # split mean_varying in S and S_bar
        mean_sub = mean[m == 1]
        mean_subcom = mean[m == 0]

        # split covariance in SS, S_barS, SS_bar and S_barS_bar
        cov_sub_sub = cov[m == 1][:, m == 1]
        cov_sub_subcom = cov[m == 1][:, m == 0]
        cov_subcom_sub = np.transpose(cov_sub_subcom)
        cov_subcom_subcom = cov[m == 0][:, m == 0]

        x_sub = np.transpose(x)
        x_sub = x_sub[self.group_mask]
        x_sub = x_sub[m == 1]
        x_sub = x_sub.reshape(-1)
        # use pseudo-inverse for inverse covariance for simplicity
        cov_sub_sub_inv = np.linalg.pinv(cov_sub_sub)

        # compute conditional mean_varying and conditional covariance
        mean_cond = np.add(mean_subcom, np.dot(cov_subcom_sub, np.dot(cov_sub_sub_inv, x_sub - mean_sub)))
        cov_cond = cov_subcom_subcom - np.dot(cov_subcom_sub, np.dot(cov_sub_sub_inv, cov_sub_subcom))

        return mean_cond, cov_cond

    def mask_mean_cov(self, mean, cov):
        """
        Exclude non-varying indices from mean and covariance because they are not important for further computations.

        :param mean: mean of given data
        :param cov: covariance of given data
        :return: mean and covariance adjusted for non-varying features
        """
        mean_varying = mean[self.group_mask]  # TODO groups can be in different order from features => make changes everywhere self.group_mask is used
        cov_varying = cov[self.group_mask][:, self.group_mask]
        return mean_varying, cov_varying
