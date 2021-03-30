# [26.11.20] OV
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
#print(len(list(accuracies_per_category.keys())))

################################################################################
#%% Extract softmax vectors of the best frame from each video per category
################################################################################
# Load categories from txt file
l_categories = utils.load_categories()
l_avg_softvectors = []
l_labels = []

start = time.time()
N = len(l_categories)
i = 0
for category in l_categories[:N]:
    #print(i, category)
    i+=1
    
    # Extract files per category
    l_files = list(accuracies_per_category[category].keys())
    
    if not l_files: # Check if any files present
        continue
    else:
        # Collect average accuracies on the TRUE category per file
        l_softvectors = []
        #print(category)
        for file_name in l_files:
            # Determine the best frame
            best, worst = utils.best_worst(accuracies_dict=accuracies_per_category,
                            category_name=category,
                            video_fname=file_name)
            # Read softmax vector at best frame -> (n_categories, 1)
            per_frame_accuracies = np.array(
                accuracies_per_category[category][file_name])[best[0]]
            # Append into list
            l_softvectors.append(per_frame_accuracies)

        # Convert l_softvectors to array:
        l_softvectors = np.array(l_softvectors)
        #print(l_softvectors.shape)
        # Dump the mean softmax vector per category into l_avg_softvectors
        l_avg_softvectors.append(l_softvectors.mean(axis=0))
        l_labels.append(str(category))

# Convert l_avg_softvectors to an array of softmax features
features_softmax = np.array(l_avg_softvectors)
stop = time.time()
duration = stop-start
print(f'Time spent: {duration:.2f}s (~ {duration/N:.4f}s per file)')
print(f'\n{len(l_avg_softvectors)} videos selected!')

#%% Save features 
print(features_softmax.shape)
path_to_output = Path('outputs/')

df = pd.DataFrame(data=features_softmax)
df.to_csv(path_to_output/'RN50_features_softmax_average.csv')

#%%
path_to_csv = 'outputs/rdm_average_softmax_cosine.csv'
df = pd.read_csv(path_to_csv)

#%% Load  Matrix adn Labels
labels= list(df.columns)[1:]
metric = 'cosine'
rdm_original = df.values[:,1:307]

################################################################################
# %% Self-similarity matrix
################################################################################


metric = 'cosine' #'euclidean'
rdm_original = squareform(pdist(features_softmax, metric=metric))
print(rdm_original.shape)

#%% Save distance matrix
path_to_output = Path('outputs/')

df = pd.DataFrame(data=rdm_original,
                  columns=l_labels)
df.to_csv(path_to_output/'rdm_average_softmax_cosine.csv')

#np.savetxt(path_to_output/'rdm_average_softmax_cosine.csv', rdm_original,
#           delimiter=',')

#%% Load csv
import pandas as pd
path_to_csv = 'outputs/rdm_average_softmax_cosine.csv'
df = pd.read_csv(path_to_csv)

#%% Load  Matrix adn Labels
labels= list(df.columns)[1:]
metric = 'cosine'
rdm_original = df.values[:,1:307]

# %% Plot Matrix
################################################################################
%matplotlib qt
# Set up the matplotlib figur
f, ax = plt.subplots(figsize=(11, 9))

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
# %% Plot Matrix
################################################################################
%matplotlib qt
# Set up the matplotlib figur
f, ax = plt.subplots(figsize=(11, 9))
labels= l_labels

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

#%%
from scipy.cluster.hierarchy import ward, fcluster
linked = linkage(rdm_original, 'ward')

x = fcluster(linked, t=, criterion='distance')
print(np.unique(x)[-1])
################################################################################
# %% Hierarchical Clustering Dendrogram (Simple)
################################################################################
linked = linkage(rdm_original, 'ward')

labelList = labels

plt.figure(figsize=(20, 5))
d = dendrogram(
            linked,
            orientation='top',
            labels=labelList,
            truncate_mode='level',
            p=38,
            distance_sort='descending',
            show_leaf_counts=False,
            leaf_rotation=45,
            color_threshold=1.5,
            above_threshold_color='gray',
            show_contracted=True,
            leaf_font_size=4.2
          )
axes = plt.gca()
axes.set_ylim([0,6])
axes.set_xticklabels(axes.get_xticklabels(), ha='right')
plt.suptitle('Clustering Dendrogram of Cosine DM from Average BestFrame per Category')
plt.tight_layout()
plt.show()
print(len(axes.get_xticklabels()))

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

"""
plt.figure(figsize=(20, 5))
utils.plot_dendrogram(model, truncate_mode='level', p=45,
                      labels=labels,
                      leaf_rotation=45,
                      #color_threshold=silhouette_coef,
                      leaf_font_size=4.2,
                      distance_sort='descending')
axes = plt.gca()
plt.tight_layout()
print(len(axes.get_xticklabels()))
"""
print(model.n_clusters)

#%% Linkage matrix (manually computed)
# Manually compute linkage matrix, 'cause sklearn's AggClust does not return
# model.distances_ when the n_clusters is defined
distances, weights = utils.get_distances(rdm_original, model, 'max')

Z = np.column_stack([model.children_, distances, weights]).astype(float)

#%% Plot clustering dendrogram 
plt.figure(figsize=(5,20))
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

#%%
clust_colors = ['C1', 'C1', 'C1']
for col in R['color_list']:
    i = int(col[1:]) + 1
    clust_colors.append('C' + str(i))

#%% Cophenetic distance
# https://journals.plos.org/plosone/article/figure?id=10.1371/journal.pone.0100782.g007
# 
c, d = cophenet(Z, squareform(rdm_original))

f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(squareform(d), ax=ax, cmap="GnBu", #linewidths=0.01,
            cbar_kws={'label': 'cophenetic distance'},
            square=True,
            xticklabels=labels,
            yticklabels=labels)

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
sns.barplot(x = x,
            y = y)
plt.title('Cophenetic correlation coefficients for different HC linkages')
plt.show()

#%% Perform HierClust with the best n_clusters
model = cluster.AgglomerativeClustering(n_clusters=l_silhouette[np.array(l_silhouette)[:, 1].argmax(axis=0)][0],
                                        linkage='ward',
                                        distance_threshold=None)
model.fit(rdm_original)

"""
plt.figure(figsize=(20, 5))
utils.plot_dendrogram(model, truncate_mode='level', p=45,
                      labels=labels,
                      leaf_rotation=45,
                      #color_threshold=silhouette_coef,
                      leaf_font_size=4.2,
                      distance_sort='descending')
axes = plt.gca()
plt.tight_layout()
print(len(axes.get_xticklabels()))
"""
print(model.n_clusters)

#%% Perform HierClust with the best n_clusters
model = cluster.AgglomerativeClustering(n_clusters=l_silhouette[np.array(l_silhouette)[:, 1].argmax(axis=0)][0],
                                        affinity='precomputed',
                                        linkage='complete',
                                        distance_threshold=None)
model.fit(rdm_original)

print(model.n_clusters)
# %%
