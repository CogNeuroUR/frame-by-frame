# [26.01.21] OV
################################################################################
# Imports
################################################################################
#%% Imports
from __future__ import print_function
# Image/video manipulation
import decord
decord.bridge.set_bridge('native')
from decord import VideoReader
from decord import cpu
from PIL import Image, ImageChops
import cv2
import ffmpeg
# Path and sys related
import os
import sys
import pickle
import time
import subprocess
from pathlib import Path
import re
# Data types & numerics
import numpy as np
import pandas as pd
# Plots
import seaborn as sns
import matplotlib
## Change matplotlib backend to Qt
%matplotlib qt
# "%" specifies magic commands in ipython
import matplotlib.pyplot as plt

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
dict_path = path_prefix / 'temp/full/accuracies_per_category_full_mitv1.pkl'
# Load from file
f = open(dict_path, 'rb')
accuracies_per_category = pickle.load(f)
# Print the categories
#print(len(list(accuracies_per_category.keys())))

#%% ############################################################################
# Extract best frame index from all available videos
################################################################################
# Load categories from txt file
l_categories = utils.load_categories()
l_best = []

N = len(l_categories)
i = 0
for category in l_categories[:N]:
    print(i, category)
    i+=1
    # Get index of the category from l_categories (from 0 to 338)
    c_idx = [i for i in range(len(l_categories))
                if l_categories[i] == category][0]
    
    # draw random file from category
    l_files = list(accuracies_per_category[category].keys())
    
    if not l_files:
        continue
    else:
        for file_name in l_files:
            best, worst= utils.best_worst(accuracies_dict=accuracies_per_category,
                                           category_name=category,
                                           video_fname=file_name)
            l_best.append([category, file_name, best[0]])

l_sorted_best = sorted(l_best)
print(l_sorted_best[:10])
 
#%% ############################################################################
# Categories with missing data (less than 3 clips)
################################################################################
# %%
input_path = Path(f'input_data/MIT_sampleVideos_RAW_WORK_IN_PROGRESS/')
l_files = list(input_path.rglob("*.mp4"))
l_files = sorted(l_files)

l_count_files = []
for file in l_files[:10]:
  print(str(file).split('/')[-2:])

#%%
l_categories = sorted(utils.load_categories())
#data_path = Path('input_data/MIT_sampleVideos_RAW_WORK_IN_PROGRESS/')
data_path = Path('input_data/GIFs/MIT_GIFs_25FPS_480x360p_1.0s_TOP-3-PER-CAT')
N = len(l_categories)

i = 0
l_count_files_all = []
l_count_missing = []
l_empty = []
for category in l_categories[:N]:
    print(i, category)
    category_path = data_path / category
    if os.path.exists(category_path):
      l_files = list(category_path.rglob("*.mp4"))
      #print('.mp4-s found: ', len(l_files))
      if len(l_files) != 0:
        l_count_files_all.append([category, len(l_files)])
        if len(l_files) < 3:
          l_count_missing.append([category, len(l_files)])
      else:
        l_empty.append(category)
    i += 1
    
print(f'{len(l_count_missing)} categories have missing video clips!')
print(f'{len(l_empty)} empty categories!')
#%% Save to csv
df_missing = pd.DataFrame(l_count_missing, columns=['Categories', 'Number of clips'])
df_full = pd.DataFrame(l_count_files_all, columns=['Categories', 'Number of clips'])
print(df_missing)
#%% Save to csv
df_missing.to_csv('outputs/20210126_RAW_WORK_IN_PROGRESS_missing_data_sorted.csv')
df_full.to_csv('outputs/20210126_RAW_WORK_IN_PROGRESS_video_count_per_category_sorted.csv')
