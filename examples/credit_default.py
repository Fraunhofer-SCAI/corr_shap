from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rivapy.sample_data.market_data.credit_default as credit_default
# Use RiVaPy repo from https://github.com/RIVACON/RiVaPy

import shap
from corr_shap import CorrExplainer

# generate credit default dataset with high correlation between savings and income
cred = credit_default.CreditDefaultData2()
cov = np.array([[1.0, 0., 0., 0.],
                [0., 1.0, 0.95, 0.],
                [0., 0.95, 1.0, 0.],
                [0., 0., 0., 1.0], ])
data = cred.sample(5, 1000, cov=cov)
data2 = data.drop(columns=['default_prob', 'defaulted'])
feature_names = ['age', 'income', 'savings', 'credit_income_ratio', 'economic_factor']

training_data, test_data = train_test_split(data2, test_size=0.2, random_state=0)
x_train = pd.DataFrame(training_data, columns=feature_names)
x_test = pd.DataFrame(test_data, columns=feature_names)


# define a model
def predict(X):
    """ Returns probability of default based on 'age', 'credit_income_ratio', 'income', and 'economic_factor',
     while ignoring 'savings'.  """
    age = X[:,0]
    age = 5.0*(1.0-age)*age
    credit_income_ratio = X[:, 3]
    credit_income_ratio = -5.0*0.3*credit_income_ratio
    income = X[:, 1]
    economic_factor = X[:, 4]
    x1 = 1.5*(income)**2
    x2 = 1.0-economic_factor
    return 1.0/(1.0+np.exp(2.0*(age+credit_income_ratio + x1+x2)))


# Default explainer
ex_default = CorrExplainer(predict, x_train, sampling="default")
shap_default = ex_default(x_test)
plt.title("Original method")
shap.summary_plot(shap_default, show=False, feature_names=x_train.columns)
plt.figure()

# Gauss explainer
ex_gauss = CorrExplainer(predict, x_train, sampling="gauss")
shap_gauss = ex_gauss(x_test)
plt.title("Gauss method")
shap.summary_plot(shap_gauss, show=False, feature_names=x_train.columns)
plt.figure()

# Copula explainer
ex_copula = CorrExplainer(predict, x_train, sampling="copula")
shap_copula = ex_copula(x_test)
plt.title("Copula method")
shap.summary_plot(shap_copula, show=False, feature_names=x_train.columns)
plt.figure()

# Empirical explainer
ex_empirical = CorrExplainer(predict, x_train, sampling="empirical")
shap_empirical = ex_empirical(x_test)
plt.title("Empirical method")
shap.summary_plot(shap_empirical, show=False, feature_names=x_train.columns)

plt.show()