from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

import shap
from corr_shap import CorrExplainer

# Choose dataset
x, y = shap.datasets.adult(n_points=1000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=0)

# Train model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# Default explainer
ex_default = CorrExplainer(model.predict, x_train, sampling="default")
shap_default = ex_default(x_test)
plt.title("Original method")
shap.summary_plot(shap_default, show=False, feature_names=x_train.columns)
plt.figure()

# Gauss explainer
ex_gauss = CorrExplainer(model.predict, x_train, sampling="gauss")
shap_gauss = ex_gauss(x_test)
plt.title("Gauss method")
shap.summary_plot(shap_gauss, show=False, feature_names=x_train.columns)
plt.figure()

# Copula explainer
ex_copula = CorrExplainer(model.predict, x_train, sampling="copula")
shap_copula = ex_copula(x_test)
plt.title("Copula method")
shap.summary_plot(shap_copula, show=False, feature_names=x_train.columns)
plt.figure()

# Empirical explainer
ex_empirical = CorrExplainer(model.predict, x_train, sampling="empirical")
shap_empirical = ex_empirical(x_test)
plt.title("Empirical method")
shap.summary_plot(shap_empirical, show=False, feature_names=x_train.columns)

plt.show()