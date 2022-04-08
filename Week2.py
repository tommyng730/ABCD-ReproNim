import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"].astype(np.float64) #3 classes of flowers.

svm_clf = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=10, loss="hinge")),])
svm_clf.fit(X, y)
scores_pipe = cross_validate(svm_clf, X, y)["test_score"]
print("feature selection on train set:", scores_pipe)

import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
import pandas as pd

chd = datasets.fetch_california_housing()
X, y = chd["data"], chd["target"]

max_depth = 3
dt_reg = DecisionTreeRegressor(max_depth=max_depth)
scores_pipe = cross_validate(dt_reg, X, y)["test_score"]
print("Cross-validate DecisionTreeRegressor:", scores_pipe)

n_estimators = 10
rf_reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
scores_pipe = cross_validate(rf_reg, X, y)["test_score"]
print("Cross-validate RandomForestRegressor:", scores_pipe)

ridge_model = Ridge(alpha=0.01)
ereg = VotingRegressor(estimators=[('rf', rf_reg), ('dt', dt_reg), ('r', ridge_model)])
scores_pipe = cross_validate(ereg, X, y)["test_score"]
print("Cross-validate Ensemble             :", scores_pipe)

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Create training pipeline
pipeline = Pipeline(steps=[('preprocessor', StandardScaler()),
                           ('regressor', LogisticRegression(solver='lbfgs', multi_class='auto'))])

scores_pipe = cross_validate(pipeline, X, y)["test_score"]
print("Cross-validate Ensemble: ", scores_pipe)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(solver='lbfgs', multi_class='auto')
clf2 = RandomForestClassifier(n_estimators=50)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

pipeline = Pipeline(steps=[('preprocessor', StandardScaler()),
                           ('regressor', eclf)])


# fit the pipeline to train a linear regression model on the training set
scores_pipe = cross_validate(pipeline, X, y)["test_score"]
print("Cross-validate Ensemble: ", scores_pipe)

from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(random_state=1)
X_2, y_2 = make_moons(n_samples=200, noise=0.05, random_state=0)

kmeans_blob = KMeans(n_clusters=3)
kmeans_moon = KMeans(n_clusters=2)
kmeans_blob.fit(X)
kmeans_moon.fit(X_2)

print(kmeans_blob.labels_)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_blob.labels_, s=60)
plt.show()

#Each cluster is defined solely by its center, which means that each cluster is a convex shape.
#As a result of this is that k-Means can only capture relatively simple shapes.
#k-Means also assumes that all clusters have the same “diameter” (drawing cluster boundaries exactly in the middle between the cluster centers).

print(kmeans_moon.labels_)
plt.scatter(X_2[:, 0], X_2[:, 1], c=kmeans_moon.labels_, s=60)
plt.show()


from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
labels = gmm.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)


from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons
from sklearn.manifold import SpectralEmbedding

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

embedding = SpectralEmbedding(n_components=2)
transformed = embedding.fit_transform(X)
print(transformed.shape)

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
labels = gmm.fit(X).predict(X)
plt.scatter(transformed[:, 0], transformed[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

gmm16 = GaussianMixture(n_components=16, covariance_type='full')
gmm16.fit(X)

X_gen, y_gen = gmm16.sample(n_samples=200)
plt.scatter(X_gen[:, 0], X_gen[:, 1]);


import scipy.io
import pandas as pd
import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.mixture import GaussianMixture

#Using AAL, it is anatomically defined, as opposed to functionally defined like most other atlases in the file.
mat = scipy.io.loadmat('acq-64dir_space-T1w_desc-preproc_space-T1w_msmtconnectome.mat')

#obtain the streamline count weighted by both SIFT and inverse node volumes
connectivity = mat["aal116_sift_invnodevol_radius2_count_connectivity"]

con = np.asarray(connectivity)
print(con.shape)

embedding = SpectralEmbedding(n_components=2)
transformed = embedding.fit_transform(con)
print(transformed.shape)

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
labels = gmm.fit(transformed).predict(transformed)
plt.scatter(transformed[:, 0], transformed[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
