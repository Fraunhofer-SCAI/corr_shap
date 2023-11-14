from scipy.special import binom
import numpy as np
import pandas as pd
import logging
import copy
import itertools
import typing
from scipy import sparse
import warnings

from shap.utils._legacy import convert_to_instance, match_instance_to_data, IdentityLink
from shap.explainers._explainer import Explainer
try:
    from shap.explainers._kernel import KernelExplainer
except ImportError:
    from shap.explainers._kernel import Kernel as KernelExplainer

from corr_shap.sampling.SamplingStrategy import SamplingStrategy
from corr_shap.sampling.sampling_factory import get_sampling_strategy

log = logging.getLogger('corr_shap')


class CorrExplainer(KernelExplainer):
    """Uses the modified Kernel SHAP method to explain the output of any function.

    The modifications (based on the paper 'Explaining individual predictions when features are dependent:
    More accurate approximations to Shapley values' by Kjersti Aas, Martin Jullum and Anders Løland)
    offer the possibility to include dependencies between features.
    There are 3 different approaches, which are described in the following sampling strategies.
    """

    def __init__(self, model, data, link=IdentityLink(), sampling: typing.Union[str, SamplingStrategy]="default", sampling_kwargs={}, **kwargs):
        """
        Creates an explainer object based on the KernelExplainer from the Kernel SHAP method
        designed for dense data with better performance.
        The sampling strategy 'default' produces the same results as the original KernelExplainer,
        the other sampling strategies are based on different methods
        that allow to take dependencies between features into account.

        :param model: function or iml.Model
            User supplied function that takes a matrix of samples (# samples x # features) and
            computes the output of the model for those samples. The output can be a vector
            (# samples) or a matrix (# samples x # model outputs).
        :param data: numpy.array or pandas.DataFrame
            The background dataset to use for integrating out features. To determine the impact
            of a feature, that feature is set to "missing" and the change in the model output
            is observed. Since most models aren't designed to handle arbitrary missing data at test
            time, we simulate "missing" by replacing the feature with the values it takes in the
            background dataset (in sampling strategy 'default') or the transformed/adjusted background dataset.
            Note: sparse case is accepted but converted to numpy.array
        :param link: "identity" or "logit"
            A generalized linear model link to connect the feature importance values to the model
            output. Since the feature importance values, phi, sum up to the model output, it often makes
            sense to connect them to the output with a link function where link(output) = sum(phi).
            If the model output is a probability then the LogitLink link function makes the feature
            importance values have log-odds units.
        :param sampling: "default" or "gauss" or "copula" or "empirical" or "gauss+empirical" or "copula+empirical"
            String that selects the sampling strategy.
            The sampling strategy 'default' produces the same results as the original KernelExplainer,
            the other sampling strategies are based on different methods
            that allow to take dependencies between features into account.
        """
        # convert data to numpy.array since methods cannot handle sparse data
        if sparse.issparse(data):
            warnings.warn("Sparse data is not supported. Data will be converted to dense dataset.")
            data = data.toarray()

        Explainer.__init__(self, model=model, **kwargs)

        super().__init__(model, data, link=link, **kwargs)

        if isinstance(sampling, str):
            self.sampling_strategy = get_sampling_strategy(sampling, self, sampling_kwargs)
        else:
            self.sampling_strategy = sampling
        if self.feature_names is None:
            self.feature_names = ["Feature " + str(i) for i in range(data.shape[1])]


    def shap_values(self, X, **kwargs):
        """Wrapper for computing shapley values, since methods cannot handle sparse data"""
        # convert data to numpy.array since methods cannot handle sparse data
        if sparse.issparse(X):
            warnings.warn("Sparse data is not supported. Data will be converted to dense dataset.")
            X = X.toarray()
        return super().shap_values(X, **kwargs)

    def explain(self, incoming_instance, **kwargs):
        """Determines for an incoming instance the contribution of the included features.

        :param incoming_instance: instance to be explained with shapley values
        :return: vector of shapley values for instance x
        """

        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        if self.data.groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
            self.M = len(self.varyingFeatureGroups)
            groups = self.data.groups
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if self.varyingFeatureGroups and all(len(groups[i]) == len(groups[0]) for i in self.varyingInds):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()

        self.sampling_strategy.set_varying_feature_groups(self.varyingFeatureGroups)
        self.sampling_strategy.set_instance(instance)

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values
        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto")

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2 ** 11

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2 ** 30
            if self.M <= 30:
                self.max_samples = 2 ** self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate()

            # weight the different subset sizes
            num_subset_sizes = int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            log.debug("weight_vector = {0}".format(weight_vector))
            log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
            log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            log.debug("M = {0}".format(self.M))

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype='int64')
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes: nsubsets *= 2
                log.debug("subset_size = {0}".format(subset_size))
                log.debug("nsubsets = {0}".format(nsubsets))
                log.debug("self.nsamples*weight_vector[subset_size-1] = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1]))
                log.debug("self.nsamples*weight_vector[subset_size-1]/nsubsets = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets))

                # see if we have enough samples to enumerate all subsets of this size
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes: w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        self.addsample(mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.addsample(mask, w)
                else:
                    break
            log.info("num_full_subsets = {0}".format(num_full_subsets))

            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug("samples_left = {0}".format(samples_left))
            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[:num_paired_subset_sizes] /= 2  # because we draw two samples each below
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info("remaining_weight_vector = {0}".format(remaining_weight_vector))
                log.info("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
                ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos]  # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.addsample(mask, 1.0)
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.addsample(mask, 1.0)
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted
                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info("weight_left = {0}".format(weight_left))
                self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()

            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            for d in range(self.D):
                vphi, vphi_var = self.solve(self.nsamples / self.max_samples, d)
                phi[self.varyingInds, d] = vphi
                phi_var[self.varyingInds, d] = vphi_var

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            phi_var = np.squeeze(phi_var, axis=1)

        return phi

    def addsample(self, m, w):
        """
        Determine samples depending on given sampling strategy
        for given subset (mask) and weights and saves them as properties.

        :param m: given mask of subset
        :param w: weights assigned to data and indicating the importance for the explanation of the instance
        """
        offset = self.nsamplesAdded * self.N
        samples, weights = self.sampling_strategy.sample(m)
        self.synth_data[offset:offset + self.N] = samples
        self.weights[offset:offset + self.N] = weights
        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def varying_groups(self, x):
        """
        Determine groups of features that differ from instance to be explained in at least one instance.

        :param x: given instance to be explained
        :return: indices of the features that differ in at least one instance of the data set
                    from the instance to be explained are divided into groups
        """

        # go over all nonzero columns in background and evaluation data
        # if both background and evaluation are zero, the column does not vary
        varying_indices = np.unique(np.union1d(self.data.data.nonzero()[1], x.nonzero()[1]))
        remove_unvarying_indices = []
        for i in range(0, len(varying_indices)):
            varying_index = varying_indices[i]
            # now verify the nonzero values do vary
            data_rows = self.data.data[:, [varying_index]]
            nonzero_rows = data_rows.nonzero()[0]

            if nonzero_rows.size > 0:
                background_data_rows = data_rows[nonzero_rows]
                num_mismatches = np.sum(np.abs(background_data_rows - x[0, varying_index]) > 1e-7)
                # Note: If feature column non-zero but some background zero, can't remove index
                if num_mismatches == 0 and not \
                    (np.abs(x[0, [varying_index]]) > 1e-7 and len(nonzero_rows) < data_rows.shape[0]):
                    remove_unvarying_indices.append(i)
        mask = np.ones(len(varying_indices), dtype=bool)
        mask[remove_unvarying_indices] = False
        varying_indices = varying_indices[mask]
        return varying_indices

    def allocate(self):
        """Prepares space for the data needed to determine the contributions of the instance.
        """
        super().allocate()
        self.weights = np.empty(self.nsamples * self.N)

    def run(self):
        """Solves the created system whose solution are the shapley values"""
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
        data = self.synth_data[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :]
        if self.keep_index:
            index = self.synth_data_index[self.nsamplesRun * self.N:self.nsamplesAdded * self.N]
            index = pd.DataFrame(index, columns=[self.data.index_name])
            data = pd.DataFrame(data, columns=self.data.group_names)
            data = pd.concat([index, data], axis=1).set_index(self.data.index_name)
            if self.keep_index_ordered:
                data = data.sort_index()
        active_indices = self.weights != 0
        modelOut = self.model.f(data[active_indices])
        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        self.y[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :][active_indices] = np.reshape(modelOut, (
            np.sum(active_indices), self.D))

        weights = self.weights.reshape((self.nsamples, self.N))
        y = self.y.reshape((self.nsamples, self.N, self.D))  # D = num dimensions of model output
        eyVal = y * weights[..., None]
        self.ey = np.sum(eyVal, axis=1)

        self.nsamplesRun = self.nsamplesAdded
