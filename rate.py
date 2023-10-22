from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  SMOTE
sm = SMOTE(sampling_strategy='auto',random_state=42,k_neighbors=5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_sm = scaler.fit_transform(X_train_sm)
X_test_sm = scaler.transform(X_test)

#  borderline-SMOTE-1
bsmote1 = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, kind='borderline-1', random_state=42)
X_bsmote1, y_bsmote1 = bsmote1.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_bsmote1 = scaler.fit_transform(X_bsmote1)
X_test_bs1 = scaler.transform(X_test)


#  borderline-SMOTE-2
bsmote2 = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, kind='borderline-2', random_state=42)
X_bsmote2, y_bsmote2 = bsmote2.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_bsmote2 = scaler.fit_transform(X_bsmote2)
X_test_bs2 = scaler.transform(X_test)


svm = SVC()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()



print('\tМетрики  SMOTE ')
svm_scores = cross_val_score(svm, X_test_sm, y_test, cv=7)
print(f"SVM scores:", np.round(svm_scores.mean(),4))


knn_scores = cross_val_score(knn, X_test_sm, y_test, cv=7)
print(f"KNN scores:" ,np.round(knn_scores.mean(),4))


rf_scores = cross_val_score(rf, X_test_sm, y_test, cv=7)
print(f"RF scores:" ,np.round(rf_scores.mean(),4))

print('\tМетрики  SMOTE 1')
svm_bsscores = cross_val_score(svm, X_bsmote1, y_bsmote1 , cv=7)
print(f"SVM scores:", np.round(svm_bsscores.mean(),4))


knn_bsscores = cross_val_score(knn, X_bsmote1, y_bsmote1 , cv=7)
print(f"KNN scores:",np.round(knn_bsscores.mean(),4))


rf_bsscores = cross_val_score(rf, X_bsmote1, y_bsmote1 , cv=7)
print(f"RF scores:",np.round(rf_bsscores.mean(),4))

print('\tМетрики  SMOTE 2')

svm_bs2scores = cross_val_score(svm, X_bsmote2, y_bsmote2 , cv=7)
print(f"SVM scores:" ,np.round(svm_bs2scores.mean(),4))

knn_bs2scores = cross_val_score(knn, X_bsmote2, y_bsmote2 , cv=7)
print(f"KNN scores:", np.round(knn_bs2scores.mean(),4))

rf_bs2scores = cross_val_score(rf, X_bsmote2, y_bsmote2 , cv=7)
print(f"RF scores:",np.round(rf_bs2scores.mean(),4))




