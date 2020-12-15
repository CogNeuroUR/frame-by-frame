# [30.10.2020] OV
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

#%% ############################################################################
# Extract best and worst frame index from all available videos
################################################################################
# Load categories from txt file
import time

l_categories = utils.load_categories()
l_best_worst = []

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
            best, worst = utils.best_worst(accuracies_dict=accuracies_per_category,
                                           category_name=category,
                                           video_fname=file_name)
            l_best_worst.append([category, file_name, best[0], worst[0]])


#%% ############################################################################
# Collect frames by the extracted indexes
################################################################################
import cv2
start = time.time()

i = 0
for category, file_name, best, worst in l_best_worst:
    # Verbose
    print(f'{i}/{len(l_best_worst)}')
    i+=1
    # Load video file using decord
    path_2_file = Path(f'data/MIT_sampleVideos_RAW/{category}/{file_name}')
    vr = VideoReader(str(path_2_file))
    # Extract best and worst frames by their label
    best_frame = vr.get_batch([best]).asnumpy()[0]
    worst_frame = vr.get_batch([worst]).asnumpy()[0]
    
    # Save to files
    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(f'.extracted/{category}', exist_ok=True)
    
    best_path = f'.extracted/{category}/{file_name}_best_{best}.jpg'
    worst_path = f'.extracted/{category}/{file_name}_worst_{worst}.jpg'
    
    cv2.imwrite(best_path, cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(worst_path, cv2.cvtColor(worst_frame, cv2.COLOR_RGB2BGR))
    
stop = time.time()
duration = stop-start
print(f'Time spent: {duration:.2f}s (~ {duration/N:.2f}s per file)')
