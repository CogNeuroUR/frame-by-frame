# [30.03.21] OV
# Script for extraction of Softmax vectors and RDM computation,
# given the softmax accuracies per videoframe 

################################################################################
# Imports
################################################################################
#%% Imports
from __future__ import print_function
# Path and sys related
import pickle
from pathlib import Path
# Data types & numerics
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import squareform
# Plots
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Set matplotlib's backend: qt : for Qt creating a separate window for each plot
# Otherwise set it to inline: '%matplotlib inline'
%matplotlib inline

#%% Import custom utility functions
import sys, importlib
sys.path.insert(0, '..')
import utils
importlib.reload(utils) # Reload If modified during runtime

# %% Load full accuracy dictionary extracted w/ ResNet50-MiTv1
# AB: extracted with '01.mif_extraction_exploratory.ipynb' or '01.mif_extraction_short.ipynb'
path_prefix = Path('..').absolute()
dict_path = path_prefix / 'temp/accuracies_per_category_mitv1_fps-25.pkl'
# Load from file
f = open(dict_path, 'rb')
accuracies_per_category = pickle.load(f)

#%%#############################################################################
# Compute softmax feature vectors
################################################################################
# Load categories from txt file
# Source: http://moments.csail.mit.edu/moments_models/category_momentsv1.txt
l_categories = utils.load_categories(path_prefix / 'models/labels/category_momentsv1.txt')

features_softmax, l_labels = utils.extract_features(l_categories=l_categories,
                                              softmax_dict=accuracies_per_category)

df_features = pd.DataFrame(data=features_softmax,
                           columns=l_categories, index=l_labels)
print(df_features[:5])

#%% Save to DataFrame to csv for external use
df_features.to_csv(Path('../outputs/RN50_features_softmax_average_nammed.csv'))

#%%#############################################################################
#  RDM computation
################################################################################
# Metric to compute dissimilarity matrix: cosine, euclidean, correlation, ...
metric = 'cosine' # cosine of the angle between two feature vectors
rdm_original = squareform(pdist(df_features, metric=metric))
print(rdm_original.shape)

#%% Plot RDM
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9), num='RDM')

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(rdm_original, ax=ax, cmap="GnBu", #linewidths=0.01,
            cbar_kws={'label': f'{metric}'},
            square=True,
            xticklabels=l_labels,
            yticklabels=l_labels)

plt.yticks(fontsize=2)
plt.xticks(fontsize=2)
plt.title('Average Softmax probabilities per MiTv1 category (ResNet50) ')
plt.show()

#%% Save distance matrix
path_to_output = Path('outputs/')

df = pd.DataFrame(data=rdm_original,
                  columns=l_labels)
df.to_csv(Path(f'../outputs/rdm_average_softmax_{metric}.csv'))
