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
# Data types & numerics
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
# Plots
import seaborn as sns
import matplotlib
#%matplotlib qt
import matplotlib.pyplot as plt
from PIL import Image

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
#%% Extract average accuracy per video 
################################################################################
# Load categories from txt file
import time

l_categories = utils.load_categories()
l_best_videos = []

N = len(l_categories)
i = 0
for category in l_categories[:N]:
    #print(i, category)
    i+=1
    # Get index of the category from l_categories (from 0 to 338)
    c_idx = [i for i in range(len(l_categories))
                if l_categories[i] == category][0]
    
    # Extract files per category
    l_files = list(accuracies_per_category[category].keys())
    
    if not l_files: # Check if any files present
        continue
    else:
        # Collect average accuracies on the TRUE category per file
        l_avg_accuracies = []
        #print(category)
        for file_name in l_files:
            # Read accuracies from dictionary -> (n_frames, n_categories)
            per_frame_accuracies = np.array(accuracies_per_category[category][file_name])
            # Take the accuracies of the TRUE category -> (n_frames, 1)
            per_frame_accuracies = per_frame_accuracies[:, c_idx]
            # Dump category, file_name and mean accuracy
            l_avg_accuracies.append([category, file_name, per_frame_accuracies.mean()])
            #print(f'{file_name} : {per_frame_accuracies.mean():.3f}')

        # Extract the best video, i.e. with highest mean acc. per TRUE category
        l_best_videos.append(l_avg_accuracies[np.array(l_avg_accuracies)[:, 2].astype(float).argmax()])
        
#print(l_best_videos)
print(f'\n{len(l_best_videos)} videos selected!')

################################################################################
#%% Collect softmax probability vectors for all videos in l_best_video
################################################################################
start = time.time()

# Define empty array of size (n_best_videos, 339)
features_softmax = np.zeros((len(l_best_videos), len(l_categories)))
l_labels = []

for i in range(len(l_best_videos)):
    category, file_name, acc = l_best_videos[i]
    # Verbose
    print(f'{i}/{len(l_best_videos)}')
    # Extract best frame's index
    best, worst = utils.best_worst(accuracies_dict=accuracies_per_category,
                                category_name=category,
                                video_fname=file_name)
    # Extract frame idx from best
    idx_best = best[0]
    # Collect softmax given best frame's index and insert into our feature array
    features_softmax[i] = np.array(accuracies_per_category[category][file_name])[idx_best]
    #l_labels.append(str(category + ' (' + file_name[:15] + '....mp4)'))
    l_labels.append(str(category))
    
stop = time.time()
duration = stop-start
print(f'Time spent: {duration:.2f}s (~ {duration/N:.2f}s per file)')

################################################################################
# %% Self-similarity matrix
################################################################################
from scipy.spatial.distance import pdist, squareform

metric = 'euclidean' #'jensenshannon'
rdm_original = squareform(pdist(features_softmax, metric=metric))
print(rdm_original.shape)

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
plt.suptitle('Distance matrix of softmax classification probabilities of the best videos per category')
plt.show()

################################################################################
# %% Hierarchical Clustering Dendrogram
################################################################################
%matplotlib qt
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(rdm_original, 'ward')

labelList = l_labels

plt.figure(figsize=(20, 5))
dendrogram(
            linked,
            orientation='top',
            labels=labelList,
            truncate_mode='level',
            p=38,
            distance_sort='descending',
            show_leaf_counts=False,
            leaf_rotation=45,
            color_threshold=1.7,
            above_threshold_color='gray',
            show_contracted=True,
            leaf_font_size=4.2
          )
axes = plt.gca()
axes.set_ylim([0,6])
axes.set_xticklabels(axes.get_xticklabels(), ha='right')
plt.suptitle('Clustering Dendrogram of Euclidean DM from BestVideo-BestFrame')
plt.tight_layout()
plt.show()
print(len(axes.get_xticklabels()))

################################################################################
#%% Compute pair-wise distance matrix
################################################################################
# Compute and plot first dendrogram
%matplotlib qt
import scipy.cluster.hierarchy as sch
import pylab

# Definitions
condensed_rdm = pdist(features_softmax, metric=metric)
rdm = rdm_original
plot_labels = l_labels

# Define figure
fig = pylab.figure(figsize=(12, 10))
ax1 = fig.add_axes([0.09,0.1,0.2,0.6])

# Compute and plot 1st dendrogram.
Y = sch.linkage(condensed_rdm, method='ward')
Z1 = sch.dendrogram(Y, orientation='left')
ax1.set_xticks([])
ax1.set_yticks([])

# Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
Y = sch.linkage(condensed_rdm, method='ward') # 'single', 'average' ...
Z2 = sch.dendrogram(Y)
ax2.set_xticks([])
ax2.set_yticks([])

# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
rdm = rdm[idx1,:]
rdm = rdm[:,idx2]
im = axmatrix.matshow(rdm, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# Add labels:
axmatrix.set_xticks(range(len(plot_labels)))
axmatrix.set_xticklabels(np.asarray(plot_labels)[idx1].tolist())#, minor=False)
axmatrix.xaxis.set_label_position('top') #('bottom')
axmatrix.xaxis.tick_bottom()

pylab.xticks(rotation=-90, fontsize=2)

axmatrix.set_yticks(range(len(plot_labels)))
axmatrix.set_yticklabels(np.asarray(plot_labels)[idx2].tolist())#, minor=False)
axmatrix.yaxis.set_label_position('left') #('right')
axmatrix.yaxis.tick_right()
pylab.yticks(fontsize=2)


# Add title
fig.suptitle(f'Hierarchical Clustering ({metric})',
             fontsize=20)
#fig.tight_layout()

# Finish plot
fig.show()
#fig.savefig(path_prefix / 'plots/HierClust_BERT-Base-German_ActHierEEG_RDM_Euclid_centroid.pdf')
#fig.savefig(path_prefix / 'USE_MiT1&K400_RDM_Euclid_HierClust.png')

################################################################################
#%% HDBSCAN
################################################################################
import hdbscan
clusterer = hdbscan.HDBSCAN()
clusterer.fit(X_red)
print(clusterer.labels_)
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)

#%% HDBSCAN
################################################################################
import hdbscan
clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2)
clusterer.fit(rdm_original)
print(clusterer.labels_)
axis = clusterer.single_linkage_tree_.plot(cmap='viridis',
                                    colorbar=True)
axis.set_xticklabels(np.asarray(plot_labels)[idx2].tolist())

# %%
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())

################################################################################
#%% SpectralEmbedding
################################################################################
from sklearn import manifold, datasets
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(features_softmax)

# Visualize the clustering
def plot_clustering(X_red, labels, y, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 10})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=10)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#plt.scatter(*X_red.T)
clusterer.fit(X_red)
plot_clustering(X_red, clusterer.labels_, l_labels, 'HDBSCAN on SpectralEmbedding')
# %%
