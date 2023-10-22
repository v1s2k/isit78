import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.neighbors import KNeighborsClassifier


data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

bs = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, kind='borderline-1', random_state=42)

X_train_bs, y_train_bs = bs.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_bs = scaler.fit_transform(X_train_bs)
X_test_bs = scaler.transform(X_test)

model = KNeighborsClassifier()
model.fit(X_train_bs, y_train_bs)

y_train_pred = model.predict(X_train_bs)
accuracy_train = accuracy_score(y_train_bs, y_train_pred)
precision_train = precision_score(y_train_bs, y_train_pred)
recall_train = recall_score(y_train_bs, y_train_pred)
f1_train = f1_score(y_train_bs, y_train_pred)

y_test_pred = model.predict(X_test_bs)
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)

print('Метрики knn SMOTE 1')
print('Train Accuracy:', np.round(accuracy_train,3))
print('Train Precision:', np.round(precision_train,3))
print('Train Recall:', np.round(recall_train,3))
print('Train F1-score:', np.round(f1_train,3))

print('Test Accuracy:', np.round(accuracy_test,3))
print('Test Precision:', np.round(precision_test,3))
print('Test Recall:', np.round(recall_test,3))
print('Test F1-score:', np.round(f1_test,3))

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_test_bs)
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test, cmap=plt.cm.get_cmap("Set1", 2), alpha=0.7)
plt.title('t-SNE visualization')
plt.show()


umap_ = umap.UMAP(random_state=42)
X_umap = umap_.fit_transform(X_test_bs)
plt.figure(figsize=(10, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_test, cmap=plt.cm.get_cmap("Set1", 2), alpha=0.7)
plt.title('UMAP Visualization')
plt.show()

plt.scatter(X_test_bs[:, 0], X_test_bs[:, 1], c=y_test, cmap='rainbow')
plt.title('классы на основе выборок с известными метками классов')
plt.show()

y_pred = model.predict(X_test_bs)
plt.scatter(X_test_bs[:, 0], X_test_bs[:, 1], c=y_pred, cmap='rainbow')
plt.title('классы с метками выставленными классификатором.')
plt.show()

print('R2:', np.round(metrics.r2_score(y_test, y_pred), 2))