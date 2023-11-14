import shap
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from corr_shap import CorrExplainer

x, y = shap.datasets.california(n_points=1000)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lin_regr = linear_model.LinearRegression()
lin_regr.fit(x_train, y_train)

ex = CorrExplainer(lin_regr.predict, x_train, sampling="default")
shapley = ex(x_test)

shap.plots.beeswarm(shapley)

