# [15.12.20] OV
# Script for best & worst frame extraction, given the ResNet50 per-frame
# accuracies

################################################################################
# Imports
################################################################################
#%% Imports
from __future__ import print_function
# Decord related
import decord
decord.bridge.set_bridge('native')
from decord import VideoReader
from decord import cpu
# Path and sys related
import os
import pickle
from pathlib import Path
import time
# Data types & numerics
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics, cluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform
# Plots
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib qt

#%% Import custom utils
import sys, importlib
sys.path.insert(0, '..')
import utils
importlib.reload(utils) # Reload If modified during runtime

################################################################################
# Load full accuracy dictionary extracted w/ ResNet50-MiTv1
################################################################################
# %% Define paths
path_prefix = Path().parent.absolute()
dict_path = path_prefix / 'saved/full/accuracies_per_category_full_mitv1.pkl'
# Load from file
f = open(dict_path, 'rb')
accuracies_per_category = pickle.load(f)
# Print the categories
#print(len(list(accuracies_per_category.keys()))

#%% Load csv
path_to_csv = 'outputs/rdm_average_softmax_cosine.csv'
df = pd.read_csv(path_to_csv)

#%% Load  Matrix adn Labels
labels= list(df.columns)[1:]
metric = 'cosine'
rdm_original = df.values[:,1:307]

################################################################################
# %% Plot Matrix
################################################################################
%matplotlib qt
# Set up the matplotlib figur
f, ax = plt.subplots(figsize=(11, 9), num='RDM')

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(rdm_original, ax=ax, cmap="GnBu", #linewidths=0.01,
            cbar_kws={'label': f'{metric}'},
            square=True,
            xticklabels=labels,
            yticklabels=labels)


plt.yticks(fontsize=2)
plt.xticks(fontsize=2)
plt.suptitle('Distance matrix of average softmax classification probabilities per category')
plt.show()

################################################################################
# %% Cophenetic distance to findout the best linkage method
################################################################################
# %% Test different linkage types and select the one with max(cophenetic_dist)
linkage_types = ['ward', 'complete', 'average', 'single']
l_cophenets = []

for linkage in linkage_types:
    # Fit model on the selected linkage
    model = cluster.AgglomerativeClustering(linkage=linkage)
    model.fit(rdm_original)
    
    # Compute Z
    distances, weights = utils.get_distances(rdm_original, model, 'max')
    Z = np.column_stack([model.children_, distances, weights]).astype(float)
    
    # Compute cophenetic correlation coeff
    c, d = cophenet(Z, squareform(rdm_original))
    l_cophenets.append([c, linkage])

# %%
print(l_cophenets)
y = np.array(l_cophenets)[:, 0].astype(float)
x = np.array(l_cophenets)[:, 1]
plt.subplots(num='Cophenetic')
sns.barplot(x = x,
            y = y)
plt.title('Cophenetic correlation coefficients for different HC linkages')
plt.show()


################################################################################
# %% Hierarchical Clustering Dendrogram (enhanced w/ Silhouette's Idx)
################################################################################
# Inspired from 
# https://rstudio-pubs-static.s3.amazonaws.com/284508_1faa54c2fb684ad689eccc0bcaa3b528.html#silhouette-coefficient

# %% Silhouette's Index to find the "best" value for n_clusters
l_silhouette = []
linkage = 'single' #'complete' #'ward' gives the highest cophenetic 
for N in range(50,260):
    model = cluster.AgglomerativeClustering(n_clusters=N,
                                            linkage=linkage)
    model.fit(rdm_original)
    silhouette_coef = metrics.silhouette_score(rdm_original,
                                               labels=model.labels_,
                                               metric='precomputed')
    l_silhouette.append([N, silhouette_coef])
   
#%% Plot results
plt.subplots(num='Silhouette\'s Idx')
plt.scatter(x = np.array(l_silhouette)[:, 0],
            y=np.array(l_silhouette)[:, 1])
plt.axvline(l_silhouette[np.array(l_silhouette)[:, 1].argmax(axis=0)][0], c='r',
            label=f'max(Silhouette) = {np.amax(np.array(l_silhouette)[:, 1]):.2f} at N={50 + np.array(l_silhouette)[:, 1].argmax(axis=0)}')
plt.legend()
plt.title(f'Silhouette\'s Idx for {linkage} linkage')
plt.show()

#%% Perform HierClust with the best n_clusters
model = cluster.AgglomerativeClustering(n_clusters=l_silhouette[np.array(l_silhouette)[:, 1].argmax(axis=0)][0],
                                        #affinity='precomputed',
                                        linkage=linkage, #'ward', #'complete',
                                        distance_threshold=None)
model.fit(rdm_original)
print(model.n_clusters)

#%% Linkage matrix (manually computed)
# Manually compute linkage matrix, 'cause sklearn's AggClust does not return
# model.distances_ when the n_clusters is defined
distances, weights = utils.get_distances(rdm_original, model, 'max')

Z = np.column_stack([model.children_, distances, weights]).astype(float)

#%% Plot clustering dendrogram 
plt.figure(figsize=(5,20), num='Dendrogram')
R = dendrogram(
                Z,
                orientation='left',
                labels=labels,
                #truncate_mode='level',
                #p=38,
                distance_sort='descending',
                show_leaf_counts=False,
                #leaf_rotation=45,
                #color_threshold=1.1, # 1.7
                above_threshold_color='lightgray',
                show_contracted=True,
                leaf_font_size=3.
          )
# Align ticklabels to right
axes = plt.gca()
#axes.set_xticklabels(axes.get_xticklabels(), ha='right')
axes.tick_params(axis='x', labelsize=7)
# Set title
axes.set_title(f'Dendrogram ({linkage} linkage) for max(SI) and n_clusters = {l_silhouette[np.array(l_silhouette)[:, 1].argmax(axis=0)][0]}',
               fontsize=8)
# Set colors (not working properly)
#[t.set_color(i) for (i,t) in zip(R['color_list'],axes.xaxis.get_ticklabels())]
plt.tight_layout()
plt.show()
print(len(axes.get_yticklabels()))
