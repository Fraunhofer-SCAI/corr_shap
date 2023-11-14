import numpy as np

from .GaussStrategy import GaussStrategy


class EmpiricalStrategy(GaussStrategy):
    """ Uses the idea of kernel density estimation to modify the kernel SHAP method
    to explain the output of a function.

    A method based on the idea that data points important for explanation must be closer
    to the data point being explained. The distance is used to determine weights
    that indicate the importance for the explanation.
    """

    def __init__(self, explainer, sigma=0.1, eta=0.9, **kwargs):
        """
        Construct all necessary attributes for the EmpiricalConditionalStrategy object, especially a smoothing parameter
        used in distance computations (sigma) and a limit for the number of used rows for explanation (eta).
        """
        super().__init__(explainer)
        self.sigma = sigma
        self.eta = eta

    def sample(self, m):
        """
        Return prepared sample data.
        Determine most important samples for explanation of instance.
        Idea: The closer data and instance to be explained are the more important they are for explanation.
        Determine distance of data and instance to be explained, determine weights based on distance
        and choose most weighted data as sample.

        :param m: given mask of subset
        :return: samples with fixed masked features and normalized weights
        """

        samples = self.data.copy()
        samples = self.set_masked_features_to_instance(m, self.x, samples)

        dist_weights = self.calculate_dist(self.x, m)
        if dist_weights is None:
            # all of the samples are too far away so that all dist_weights would be 0
            # => use equal distweights for all datapoints
            dist_weights = np.ones(self.data.shape[0])

        data_weights = self.data_weights.copy()
        data_weights[dist_weights == 0] = 0
        data_weights = self.normalize(data_weights)
        weights = dist_weights * data_weights

        return samples, weights

    def normalize(self, weights):
        """ Return normalized data weights """
        sum = np.sum(weights)
        if sum != 0:
            weights = weights/sum
        return weights

    def calculate_dist(self, x, m):
        """
        Computes distance and weights for empirical distribution method.
        Based on a modified version of mahalanobis distance, weights are calculated for each row of data.
        All data rows that are important enough until a limit (eta) is reached are used as sample data.
        Weights of rows over this limit are set to 0.

        :param x: given instance to be explained
        :param m: given mask of subset
        :return: weights assigned to sample data
        """
        # preparing data for calculating distance
        subset_size = np.sum(m)
        cov_S = self.cov_varying[m == 1][:, m == 1]
        cov_S_inv = np.linalg.pinv(cov_S)
        x_S = x[0, self.group_mask == 1]
        x_S = x_S[m == 1]
        dataset_S = self.data[:, self.group_mask == 1][:, m == 1]

        x_diff_S = x_S - dataset_S
        d_S2_matrix = x_diff_S[:, :, None] * cov_S_inv[None, :, :] * x_diff_S[:, None, :]  # d_S2_matrix[i,j,k] = x_diff_S[i, j] * cov_S_inv[j, k] * x_diff_S[i, k]
        d_S2 = np.sum(np.sum(d_S2_matrix, axis=-1), axis=-1)  # d_S2[i] = sum_j sum_k d_S2_matrix[i,j,k]
        d_S2 = np.abs(d_S2/subset_size)  # distance D_S ^2
        w_S = np.exp(-d_S2 / (2 * self.sigma * self.sigma))  # weights
        w_sum = np.sum(w_S)
        if w_sum == 0:
            return None
        w_S = w_S / w_sum  # normalize weights
        sorted_ind = np.argsort(w_S)[::-1]
        w_cumsum = np.cumsum(w_S[sorted_ind])
        K = np.searchsorted(w_cumsum, self.eta, side="right")
        w_S[sorted_ind[K+1:]] = 0
        w_S = w_S / np.sum(w_S) * np.sum(w_S != 0)  # normalize again
        return w_S