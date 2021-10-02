# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 19:21:38 2021

@author: Jiaxing
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

#get data
olivetti = fetch_olivetti_faces()
X = olivetti.data
y = olivetti.target

#stratified sampling, 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

#svc
svc = SVC()
svc.fit(X_train, y_train)

cross_val_score(svc, X_train, y_train, cv=5, scoring="accuracy")
svc.score(X_train,y_train)
svc.score(X_test,y_test)

#K-Means
n_clusters = 10
cluster_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
print(n_clusters)
print(silhouette_avg)
for i in np.arange(2,n_clusters):
    cluster_labels = KMeans(n_clusters=i, random_state=42).fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(i)
    print(silhouette_avg)
    
#the best n_clusters is 2
y_cluster = KMeans(n_clusters=2, random_state=42).fit_predict(X)
X_train, X_test, y_train, y_test = train_test_split(X, y_cluster, test_size=0.2, stratify=y)

#svc using cluster data
svc_new = SVC()
svc_new.fit(X_train, y_train)

cross_val_score(svc_new, X_train, y_train, cv=5, scoring="accuracy")
svc_new.score(X_train,y_train)
svc_new.score(X_test,y_test)