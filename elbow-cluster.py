#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:34:34 2019

@author: mgermano
"""

# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from kneed.knee_locator import KneeLocator
from sklearn.preprocessing import StandardScaler,QuantileTransformer,MinMaxScaler



z_points = pd.DataFrame(np.load('./data/z_points_3L.npy')) 
slope = pd.DataFrame(np.load('./data/slope_3L.npy'))

variables = pd.concat([z_points, slope],axis=1)

variable_normal = pd.DataFrame(QuantileTransformer(output_distribution='normal').fit_transform(variables))

#variable_normal = pd.DataFrame(np.load('/Users/mgermano/Documents/PhD/clustering/temperature_approach/outliers_analysis/data_IQR-OR.npy'))

#Elbow method 
ks = range(1, 20)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(variable_normal.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
    
kn = KneeLocator(ks, inertias, curve='convex', direction='decreasing')
knee_yconvdec = kn.knee

plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.axvline(x=knee_yconvdec, color='gray', linestyle='--')
plt.xticks(ks)
plt.show()
plt.tight_layout()

# Specify the number of clusters (3) and fit the data X
kmeans = KMeans(n_clusters=5,n_init=100).fit(variable_normal)
# Get the cluster centroids
print(kmeans.cluster_centers_)
    
# Get the cluster labels
print(kmeans.labels_)
clusters = kmeans.labels_
clusters = pd.DataFrame(clusters)

#export_cluster = clusters.to_csv(r'./clusters_IQR.csv', index = None, header=True)

 


