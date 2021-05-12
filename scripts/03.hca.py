# [30.03.21] OV
# Script for performing Hierarchical Cluster Analysis, given the (Softmax) RDMs

################################################################################
# Imports
################################################################################
#%% Imports
from __future__ import print_function
# Path and sys related
from pathlib import Path
# Data types & numerics
import numpy as np
import pandas as pd
# sklearn (sci-kit learn) package contains utilities & classes for basic ML tasks
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

# Change matplotlib backend to Qt
#%matplotlib qt
# "%" specifies magic commands in ipython

#%% Import custom utils
import sys, importlib
sys.path.insert(0, '..')
import utils
importlib.reload(utils) # Reload If modified during runtime

#%%#############################################################################
# Load RDM
################################################################################
#%% Load csv
metric = 'cosine' # or 'euclidean', 'correlation', 'minkowski', etc.
path_to_csv = Path(f'../temp/outputs/rdm_average_softmax_{metric}.csv')
df_rdm = pd.read_csv(path_to_csv)
rdm_original = df_rdm.values[:,1:307]
labels= list(df_rdm.columns)[1:]

#%% Plot RDM
# AB: important visualization step
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9), num='RDM')

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(rdm_original, ax=ax, cmap="GnBu", #linewidths=0.01,
            cbar_kws={'label': f'{metric}'},
            square=True,
            xticklabels=labels,
            yticklabels=labels)

plt.yticks(fontsize=2)
plt.xticks(fontsize=2)
plt.title('Average Softmax probabilities per MiTv1 category (ResNet50) ')
plt.show()

#%%#############################################################################
# Cophenetic distance to findout the best linkage method
################################################################################
# Given a dataset X and a linkage matrix Z, the cophenetic distance between
# two points of X is the distance between the largest two distinct clusters 
# that each of the points (scipy.cluster.hierarchy.cophenet Docu).
# %% Test different linkage types and select the one with max(cophenetic_dist)
linkage_types = ['ward', 'complete', 'average', 'single']
l_cophenets = []

for linkage in linkage_types:
    # Fit model on the selected linkage
    model = cluster.AgglomerativeClustering(linkage=linkage)
    model.fit(rdm_original)
    
    # Compute distances between clusters
    distances, weights = utils.get_distances(rdm_original, model, 'max')
    Z = np.column_stack([model.children_, distances, weights]).astype(float)
    
    # Compute cophenetic correlation coeff
    c, d = cophenet(Z, squareform(rdm_original))
    l_cophenets.append([c, linkage])

print(l_cophenets)

# %% Plot cophenetic distances per linkage
y = np.array(l_cophenets)[:, 0].astype(float)
x = np.array(l_cophenets)[:, 1]
plt.subplots(num='Cophenetic')
sns.barplot(x = x,
            y = y)
plt.title('Cophenetic correlation coefficients for different HC linkages')
plt.show()

#%%#############################################################################
# Hierarchical Clustering (enhanced w/ Silhouette's Idx)
################################################################################
# Inspired from: 
# https://rstudio-pubs-static.s3.amazonaws.com/284508_1faa54c2fb684ad689eccc0bcaa3b528.html#silhouette-coefficient
# %% Silhouette's Index to find the "best" value for n_clusters
l_silhouette = []
linkage = 'ward' #'complete' #'ward' gives the highest cophenetic 
for N in range(50,260):
    model = cluster.AgglomerativeClustering(n_clusters=N,
                                            linkage=linkage)
    model.fit(rdm_original)
    silhouette_coef = metrics.silhouette_score(rdm_original,
                                               labels=model.labels_,
                                               metric='precomputed')
    l_silhouette.append([N, silhouette_coef])

#%% Plot Silhouette results
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

#%% Save clustering results as a nested list
l_nested_categories = []
temp = np.array(labels)

for i in range(max(model.labels_)):
    cat = temp[model.labels_ == i]
    x = []
    for label in cat:
        x.append(label)
        
    l_nested_categories.append(x)
print(l_nested_categories)

#%% Save as csv file
nested_clusters = pd.DataFrame(l_nested_categories)
path_to_csv = f'../temp/outputs/clusters_hc_ward_{model.n_clusters}.csv'
nested_clusters.to_csv(path_to_csv)

#%%#############################################################################
# Dendrogram Plot
################################################################################
#%% Linkage matrix (manually computed)
# Manually compute linkage matrix, 'cause sklearn's AggClust does not return
# model.distances_ when the n_clusters is defined
distances, weights = utils.get_distances(rdm_original, model, 'max')

Z = np.column_stack([model.children_, distances, weights]).astype(float)

#%% Plot clustering dendrogram 
from random import randint
colors = []
for i in range(340):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

plt.figure(figsize=(5,20), num='Dendrogram') # AB:Create visualization
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
                #leaf_font_size=3.,
                #link_color_func=lambda k: colors[k]
          )

#%%#############################################################################
# Auxiliar Cluster visualization based colouring of the dendrogram
################################################################################
# Create HTML representation of cluster colors
# For details, see https://nbviewer.jupyter.org/gist/vals/150ec97a5b7db9c82ee9
from collections import defaultdict
from matplotlib.colors import rgb2hex, colorConverter
cluster_idxs = defaultdict(list)
for c, pi in zip(R['color_list'], R['icoord']):
    for leg in pi[1:3]:
        i = (leg - 5.0) / 10.0
        if abs(i - int(i)) < 1e-5:
            cluster_idxs[c].append(int(i))

cluster_idxs

class Clusters(dict):
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = rgb2hex(colorConverter.to_rgb(c))
            html += '<tr style="border: 0;">' \
            '<td style="background-color: {0}; ' \
                       'border: 0;">' \
            '<code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>' 
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'

        html += '</table>'

        return html
    
def get_cluster_classes(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))

    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den[label][i] for i in l]
        cluster_classes[c] = i_l

    return cluster_classes

#%% Get clusters by color and print
get_cluster_classes(R)
nested_clusters = get_cluster_classes(R)

nested_clusters
#%% Collect nested list
l_nested_clusters = []

for color in nested_clusters.keys():
  l_nested_clusters.append(nested_clusters[color])
print(l_nested_clusters)

# %% Print items per cluster
for item in nested_clusters.items():
    print(item)
    print('\n')

# %%
