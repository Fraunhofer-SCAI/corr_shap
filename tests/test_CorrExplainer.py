import unittest
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

import help_functions as help_fun
from corr_shap.CorrExplainer import CorrExplainer


class ShapleyTesting(unittest.TestCase):

    def test_gauss_distributed_features(self):
        """ Test a linear model using 2000 Gauss distributed sample data
        with 6 dimensions, mean=0 and covariance of rho=0.9.
        Compute original Shapley Values and compare results.
        """
        n = 2000
        rho = 0.9

        # create data and model
        X, mean, cov = help_fun.create_input_gauss(rho, n, dim=6)
        X_train, X_test = train_test_split(X, test_size=100, random_state=0)
        model = help_fun.create_lin_model(X_train)

        # create explainers and compute shap values
        ex = shap.KernelExplainer(model.predict, X_train)
        ex_default = CorrExplainer(model.predict, X_train, sampling="default")
        ex_gauss = CorrExplainer(model.predict, X_train, sampling="gauss")
        ex_copula = CorrExplainer(model.predict, X_train, sampling="copula")
        ex_empcond = CorrExplainer(model.predict, X_train, sampling="empirical")
        ex_gauss_empirical = CorrExplainer(model.predict, X_train, sampling="gauss+empirical")
        ex_copula_empirical = CorrExplainer(model.predict, X_train, sampling="copula+empirical")

        shap_orig = ex.shap_values(X_test)
        shap_default = ex_default.shap_values(X_test)
        shap_gauss = ex_gauss.shap_values(X_test)
        shap_copula = ex_copula.shap_values(X_test)
        shap_empcond = ex_empcond.shap_values(X_test)
        shap_gauss_empirical = ex_gauss_empirical.shap_values(X_test)
        shap_copula_empirical = ex_copula_empirical.shap_values(X_test)

        # compute true shapley values
        payoff = lambda subset, x: help_fun.payoff_gauss(subset, x, mean=mean, cov=cov)
        shap_exact = help_fun.compute_exact_shapley(X_test, payoff)

        # compute error between shap values and true shapley values
        error_orig = np.abs(shap_orig - shap_exact).mean()
        error_default = np.abs(shap_default - shap_exact).mean()
        error_gauss = np.abs(shap_gauss - shap_exact).mean()
        error_copula = np.abs(shap_copula - shap_exact).mean()
        error_emp_cond = np.abs(shap_empcond - shap_exact).mean()
        error_gauss_empirical = np.abs(shap_gauss_empirical - shap_exact).mean()
        error_copula_empirical = np.abs(shap_copula_empirical - shap_exact).mean()

        print("Error Original Kernel Shap method:", error_orig)
        print("Error Default method:", error_default)
        print("Error Gauss method:", error_gauss)
        print("Error Copula method:", error_copula)
        print("Error EmpCond method:", error_emp_cond)
        print("Error CombiGauss method:", error_gauss_empirical)
        print("Error CombiCopula method:", error_copula_empirical)

        # Assert
        self.assertAlmostEqual(error_default, error_orig)
        self.assertLess(error_gauss, error_orig)
        self.assertLess(error_copula, error_orig)
        self.assertLess(error_emp_cond, error_orig)
        self.assertLess(error_gauss_empirical, error_orig)
        self.assertLess(error_copula_empirical, error_orig)

    def test_multimodal_gauss_distributed_features(self):
        """ Test a linear model using 2000 multimodal Gauss distributed sample data
        with 3 dimensions, mean1=-mean2 and covariance of rho=0.9.
        Compute original Shapley Values and compare results.
        """
        n = 2000

        # create data and model
        X, mean1, mean2, cov = help_fun.create_input_gauss_mix(numb=n, gamma=7.5, rho=0.2)
        X_train, X_test = train_test_split(X, test_size=100, random_state=0)
        model = help_fun.create_lin_model(X_train)

        # create explainers and compute shap values
        ex = shap.KernelExplainer(model.predict, X_train)
        ex_default = CorrExplainer(model.predict, X_train, sampling="default")
        ex_gauss = CorrExplainer(model.predict, X_train, sampling="gauss")
        ex_copula = CorrExplainer(model.predict, X_train, sampling="copula")
        ex_empcond = CorrExplainer(model.predict, X_train, sampling="empirical")
        ex_gauss_empirical = CorrExplainer(model.predict, X_train, sampling="gauss+empirical")
        ex_copula_empirical = CorrExplainer(model.predict, X_train, sampling="copula+empirical")

        shap_orig = ex.shap_values(X_test)
        shap_default = ex_default.shap_values(X_test)
        shap_gauss = ex_gauss.shap_values(X_test)
        shap_copula = ex_copula.shap_values(X_test)
        shap_empcond = ex_empcond.shap_values(X_test)
        shap_gauss_empirical = ex_gauss_empirical.shap_values(X_test)
        shap_copula_empirical = ex_copula_empirical.shap_values(X_test)

        # compute true shapley values
        payoff = lambda subset, x: help_fun.payoff_gauss_mix(subset, x, mean1=mean1, mean2=mean2, cov=cov)
        shap_exact = help_fun.compute_exact_shapley(X_test, payoff)

        # compute error between shap values and true shapley values
        error_orig = np.abs(shap_orig - shap_exact).mean()
        error_default = np.abs(shap_default - shap_exact).mean()
        error_gauss = np.abs(shap_gauss - shap_exact).mean()
        error_copula = np.abs(shap_copula - shap_exact).mean()
        error_emp_cond = np.abs(shap_empcond - shap_exact).mean()
        error_gauss_empirical = np.abs(shap_gauss_empirical - shap_exact).mean()
        error_copula_empirical = np.abs(shap_copula_empirical - shap_exact).mean()

        print("Error Original Kernel Shap method:", error_orig)
        print("Error Default method:", error_default)
        print("Error Gauss method:", error_gauss)
        print("Error Copula method:", error_copula)
        print("Error EmpCond method:", error_emp_cond)
        print("Error CombiGauss method:", error_gauss_empirical)
        print("Error CombiCopula method:", error_copula_empirical)

        # Assert
        self.assertAlmostEqual(error_default, error_orig)
        self.assertLess(error_gauss, error_orig)
        self.assertLess(error_copula, error_orig)
        self.assertLess(error_emp_cond, error_orig)
        self.assertLess(error_gauss_empirical, error_orig)
        self.assertLess(error_copula_empirical, error_orig)

    def test_null_model_small(self):
        """ Test a small null model .
        """
        # create model and data
        model = lambda x: np.zeros(x.shape[0])
        x_train = np.ones((2, 4))
        x_test = np.ones((1, 4))

        # create explainers and compute shap values
        ex_orig = shap.KernelExplainer(model, x_train, nsamples=100)
        ex_base = CorrExplainer(model, x_train, nsamples=100)
        e_orig = ex_orig.explain(x_test)
        e_base = ex_base.explain(x_test)

        # Assert
        error_orig = np.abs(e_orig - e_base).mean()
        print("Error between original and default method:", error_orig)
        self.assertLessEqual(error_orig, 10 ** (-5))
        self.assertLessEqual(np.abs(e_base).mean(), 10 ** (-5))

    def test_independent_linear_60(self):
        """ Test CorrExplainer with linear model and 'independent linear 60' - data
        and compare it with KernelExplainer results.
        """
        # create model and data
        x,y = shap.datasets.independentlinear60()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)

        # create explainers and shap values
        ex_original = shap.KernelExplainer(model.predict, x_train)
        shap_original = ex_original.shap_values(x_test)

        ex_dependent = CorrExplainer(model.predict, x_train, sampling="default")
        shap_dependent = ex_dependent.shap_values(x_test)

        # assert
        error_orig = np.abs(shap_original - shap_dependent).mean()
        print("Error between original and default method:", error_orig)
        self.assertLessEqual(error_orig, 10**(-5))

    def test_diabetes(self):
        """ Test CorrExplainer with linear model and 'diabetes' - data
        and compare it with KernelExplainer results.
        """
        # create model and data
        x,y = shap.datasets.diabetes()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)

        # create explainer and shap values
        ex_original = shap.KernelExplainer(model.predict, x_train)
        shap_original = ex_original.shap_values(x_test)

        ex_dependent = CorrExplainer(model.predict, x_train, sampling="default")
        shap_dependent = ex_dependent.shap_values(x_test)

        # assert
        error_orig = np.abs(shap_original - shap_dependent).mean()
        print("Error between original and default method:", error_orig)
        self.assertLessEqual(error_orig, 10**(-5))


    def test_adult(self):
        """ Test CorrExplainer with linear model and a part of the 'adult' - data
            and compare it with KernelExplainer results.
            """
        # create model and data
        x, y = shap.datasets.adult()
        x, y = x[:1000], y[:1000]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)

        # create explainers and compute shap values
        ex_original = shap.KernelExplainer(model.predict, x_train)
        shap_original = ex_original.shap_values(x_test)

        ex_dependent = CorrExplainer(model.predict, x_train, sampling="default")
        shap_dependent = ex_dependent.shap_values(x_test)

        # assert
        error_orig = np.abs(shap_original - shap_dependent).mean()
        print("Error between original and default method:", error_orig)
        self.assertLessEqual(error_orig, 10**(-5))

    def test_a1a(self):
        """ Test CorrExplainer with linear model and a part of the 'adult' - data
            and compare it with KernelExplainer results.
            """
        # create model and data
        x, y = shap.datasets.a1a()
        x, y = x[:1000], y[:1000]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)

        # create explainers and compute shap values
        ex_original = shap.KernelExplainer(model.predict, x_train)
        shap_original = ex_original.shap_values(x_test)

        ex_dependent = CorrExplainer(model.predict, x_train, sampling="default")
        shap_dependent = ex_dependent.shap_values(x_test)

        # assert
        error_orig = np.abs(shap_original - shap_dependent).mean()
        print("Error between original and default method:", error_orig)
        self.assertLessEqual(error_orig, 10**(-5))

if __name__ == "__main__":
    unittest.main()
