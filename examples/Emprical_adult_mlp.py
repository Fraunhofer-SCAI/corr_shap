import shap
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from corr_shap import CorrExplainer

x, y = shap.datasets.adult(n_points=1000)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
mlp = MLPClassifier()
mlp.fit(x_train, y_train)

ex = CorrExplainer(mlp.predict, x_train, sampling="empirical")
shapley = ex(x_test)

shap.plots.beeswarm(shapley)
