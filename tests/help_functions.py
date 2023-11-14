import numpy as np
from sklearn import linear_model
import scipy
import math
from itertools import product


def create_input_gauss(rho, numb, dim=6, mean=None):
    ''' Creates gauss distributed data of shape (numb,dim) with mean 0 and covariance rho.
    '''
    if mean is None:
        mean = np.zeros(dim)
    cov = np.ones((dim, dim)) * rho
    np.fill_diagonal(cov, 1)
    X = np.random.multivariate_normal(mean, cov, numb)
    return X, mean, cov


# creates input for mixed gauss features
def create_input_gauss_mix(gamma, numb, rho=0.2):
    ''' Creates multimodal gauss distributed data of shape (numb,3) with mean1 = - mean2 and covariance rho.
    '''
    # create input
    numb = int(numb / 2)
    vec = np.array([1, -0.5, 1])
    mean1 = gamma * vec
    mean2 = -mean1.copy()
    cov = np.array([[1, rho, rho], [rho, 1, rho], [rho, rho, 1]])
    gauss1 = np.random.multivariate_normal(mean1, cov, numb)
    gauss2 = np.random.multivariate_normal(mean2, cov, numb)
    X = np.vstack([gauss1, gauss2])

    return X, mean1, mean2, cov


# creates output using simple sampling method and creates linear regression
def create_lin_model(X_train):
    '''Creates linear model that adds up the features of each instance and adds a noise term.
    '''
    # create labels
    eps = np.random.normal(0, 0.001, X_train.shape[0])
    y_train = np.sum(X_train, axis=1) + eps
    # fit model
    lin_regr = linear_model.LinearRegression()
    lin_regr.fit(X_train, y_train)
    return lin_regr


def payoff_gauss(subset, x, mean, cov):
    '''Computes the payoff of gauss distributed data x with mean and covariance for a given subset.
    '''
    # computes payoff in simple sampling model
    # conditional mean for features that are not in S is computed equivalently to _mod_kernel
    mean_sub = np.sum(x[subset == 1])
    mean_subcom, cov = cond_mean_setting(subset, x, mean, cov)
    payoff = np.sum(mean_sub) + np.sum(mean_subcom)
    return payoff


def payoff_gauss_mix(subset, x, mean1, mean2, cov):
    '''Computes the payoff of multimodalmgauss distributed data x with mean1, mean2 and covariance for a given subset.
    '''
    mean_sub = np.sum(x[subset == 1])
    if np.sum(subset) == 0:
        return mean_sub
    mean_subcom1, cov1 = cond_mean_setting(subset, x, mean1, cov)
    mean_subcom2, cov2 = cond_mean_setting(subset, x, mean2, cov)
    m1 = scipy.stats.multivariate_normal.pdf(x[subset == 1], mean=mean1[subset == 1], cov=cov[subset == 1, subset == 1])
    m2 = scipy.stats.multivariate_normal.pdf(x[subset == 1], mean=mean2[subset == 1], cov=cov[subset == 1, subset == 1])
    mean_subcom = (m1 * mean_subcom1 + m2 * mean_subcom2) / (m1 + m2)
    payoff = mean_sub + np.sum(mean_subcom)
    return payoff


def cond_mean_setting(m, x, mean, cov):
    '''Computes conditional mean and conditional covariance for given data, mean, cov and subset.
    '''
    mean_sub = mean[m == 1]
    mean_subcom = mean[m == 0]

    # Daniela: Kovarianz aufteilen in SS, S_barS, SS_bar und S_barS_bar
    cov_sub_sub = cov[m == 1][:, m == 1]
    cov_sub_subcom = cov[m == 1][:, m == 0]
    cov_subcom_sub = np.transpose(cov_sub_subcom)
    cov_subcom_subcom = cov[m == 0][:, m == 0]

    # Daniela: Bedigten Erwartungswert und bedingte Kovarianz berechnen
    if np.linalg.det(cov_sub_sub) == 0:
        print("Nicht invertierbare Kovarianz, Berechnung auf Basis der Pseudo-Inversen")
    # Daniela: Berechnung auf Basis von der Pseudo-Inversen
    x_sub = np.transpose(x)
    x_sub = x_sub[m == 1]
    x_sub = x_sub.reshape(-1)
    cov_sub_sub_inv = np.linalg.pinv(cov_sub_sub)

    mean_cond = np.add(mean_subcom, np.dot(cov_subcom_sub, np.dot(cov_sub_sub_inv, x_sub - mean_sub)))
    cov_cond = cov_subcom_subcom - np.dot(cov_subcom_sub, np.dot(cov_sub_sub_inv, cov_sub_subcom))

    return mean_cond, cov_cond


def shapley_values(x_instance, payoff):
    '''Compute exact shapley values for an instance using calculated payoffs.
    '''
    x_instance = np.array(x_instance)
    num_of_features = x_instance.shape[0]
    phi = np.zeros(num_of_features)
    mask = np.array(list(product([0, 1], repeat=num_of_features)))
    for feature in range(num_of_features):
        combi_wout_feature = mask[mask[:, feature] == 0]
        shapley = 0
        for subset in range(combi_wout_feature.shape[0]):
            size = np.sum(combi_wout_feature[subset])
            weight = (math.factorial(size) * math.factorial(num_of_features - size - 1)) / math.factorial(
                num_of_features)
            combi_w_feature = combi_wout_feature[subset].copy()
            combi_w_feature[feature] = 1
            p1 = payoff(combi_w_feature, x_instance)
            p2 = payoff(combi_wout_feature[subset], x_instance)
            mag_contribution = p1 - p2
            shapley += weight * mag_contribution
        phi[feature] = shapley
    return phi


def compute_exact_shapley(test_data, payoff):
    '''Wrapper function to compute true shapley values for a set of instances using calculated payoffs.
        '''
    test_data = np.array(test_data)
    values = np.zeros(test_data.shape)
    if len(np.shape(test_data)) > 1:
        for line in range(test_data.shape[0]):
            values[line] = shapley_values(test_data[line], payoff)
        return values
    else:
        print(np.shape(test_data))
        exact = shapley_values(test_data, payoff)
        return exact
