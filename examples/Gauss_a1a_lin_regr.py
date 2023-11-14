import shap
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np

from corr_shap import CorrExplainer

x, y = shap.datasets.a1a()
x, y = x[:1000], y[:1000]
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[500:530], y[500:530]

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lin_regr = linear_model.LinearRegression()
lin_regr.fit(x_train, y_train)

ex = CorrExplainer(lin_regr.predict, x_train, sampling="gauss")
shapley = ex(x_test)

ex_combi = CorrExplainer(lin_regr.predict, x_train, sampling="gauss+empirical")
shapley_combi = ex_combi(x_test)

shap.plots.bar(shapley[:1])
shap.plots.bar(shapley_combi[:1])
