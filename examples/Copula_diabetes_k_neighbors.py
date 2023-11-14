import shap
from sklearn.model_selection import train_test_split
import sklearn

from corr_shap import CorrExplainer

x, y = shap.datasets.diabetes(n_points=1000)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
knn = sklearn.neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)

ex = CorrExplainer(knn.predict_proba, x_train, sampling="copula")
shapley = ex(x_test)

ex_combi = CorrExplainer(knn.predict_proba, x_train, sampling="copula+empirical")
shapley_combi = ex_combi(x_test)

shap.plots.bar(shapley[:1])
shap.plots.bar(shapley_combi[:1])


