import numpy as np


class SamplingStrategy:
    def __init__(self, explainer, **kwargs):
        """ Construct all necessary attributes for the SamplingStrategy object."""
        self.data = explainer.data.data
        self.data_weights = explainer.data.weights
        self.data_weight_sum = np.sum(self.data_weights)
        self.N = explainer.N  # num samples in self.data

    def sample(self, m):
        """
        Return prepared sample data.
        These data have fixed features for those contained in subset (m=1) and normalized weights.

        :param m: given mask of subset
        :return: samples with fixed masked features and normalized weights
        """
        x = self.x
        samples = self.data.copy()
        samples = self.set_masked_features_to_instance(m, x, samples)
        weights = self.normalize(self.data_weights)
        return samples, weights

    def normalize(self, weights):
        """ Normalize weights by their sum"""
        if self.data_weight_sum != 0:
            weights = weights/self.data_weight_sum
        return weights

    def set_masked_features_to_instance(self, m, x, samples):
        """
        Set masked features for subset to given instance.

        :param m: given mask of subset
        :param x: given instance to be explained
        :param samples: background data that are the basis for the sample
        :return: samples with fixed masked features
        """
        if isinstance(self.varyingFeatureGroups, (list,)):
            for j in range(self.varyingFeatureGroups.shape[0]):
                for k in self.varyingFeatureGroups[j]:
                    if m[j] == 1.0:
                        samples[:, k] = x[0, k]
        else:
            # for non-jagged numpy array we can significantly boost performance
            mask = m == 1.0
            groups = self.varyingFeatureGroups[mask]
            if len(groups.shape) == 2:
                for group in groups:
                    samples[:, group] = x[0, group]
            else:
                # further performance optimization in case each group has a single feature
                evaluation_data = x[0, groups]
                samples[:, groups] = evaluation_data
        return samples

    def set_instance(self, instance):
        """ Set instance to x. """
        self.x = instance.x.copy()


    def set_varying_feature_groups(self, varying_groups):
        """ Set indicies of varying feature groups."""
        self.varyingFeatureGroups = varying_groups

