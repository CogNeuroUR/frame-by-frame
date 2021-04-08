    # [20.10.20] OV
# 
# Script for building an interactive widget to present model classification
# accuracies over frames
################################################################################
# Imports
################################################################################
#%% Imports
from __future__ import print_function
# Path and sys related
import os
import pickle
from pathlib import Path

#%% Import custom utils
import sys, importlib
sys.path.insert(0, '..')
import utils
importlib.reload(utils) # Reload If modified during runtime

################################################################################
# Load partial dictionaries
################################################################################
# %% Define paths
path_prefix = Path().parent.absolute()
dict1_path = path_prefix / 'saved/full/accuracies_per_category_m50.pkl'
dict2_path = path_prefix / 'saved/full/accuracies_per_category_from50.pkl'
dict3_path = path_prefix / 'saved/full/accuracies_per_category_from100_250.pkl'
dict4_path = path_prefix / 'saved/full/accuracies_per_category_from250_.pkl'
dict5_path = path_prefix / 'saved/full/accuracies_per_category_missing.pkl'

# Load from file
f1 = open(dict1_path, 'rb')
f2 = open(dict2_path, 'rb')
f3 = open(dict3_path, 'rb')
f4 = open(dict4_path, 'rb')
f5 = open(dict5_path, 'rb')

#%% Load files and dump into dictionary
accuracies_per_category = pickle.load(f1)
accuracies_per_category.update(pickle.load(f2))
accuracies_per_category.update(pickle.load(f3))
accuracies_per_category.update(pickle.load(f4))
accuracies_per_category.update(pickle.load(f5))

# Print the categories
print(len(list(accuracies_per_category.keys())))

################################################################################
# Check for missing categories, if any
################################################################################
# If there are missing categories, run an additional round of frame extraction
# ONLY on the missing categories, using the video_frame_extractor.ipynb
#%% Load categories list
l_categories = utils.load_categories()

#%% Check for missing categories
print('Missing categories:')
for category in l_categories:
    if category not in list(accuracies_per_category.keys()):
        print(category)
    #print(len(list(accuracies_per_category.keys())))

#%% Save back a full dictionary file:
dict_path = path_prefix / 'saved/full/accuracies_per_category_full_mitv1.pkl'
f = open(dict_path, 'wb')
pickle.dump(accuracies_per_category, f)
f.close()