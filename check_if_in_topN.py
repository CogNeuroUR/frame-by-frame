# [23.10.20] OV
# Script for investigation of per-frame TopN accuracies extracted using an MiTv1
# pretrained ResNet50

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
# Data types
import numpy as np
import pandas as pd
# Plots
import seaborn as sns
import matplotlib
#%matplotlib qt
import matplotlib.pyplot as plt
from PIL import Image
# Widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
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
# Define category and file to be examined
################################################################################
#%% Define example file
# category and file names
N = 5
c_name = 'chopping' # because it's loadead in repo
f_name =  'flickr-3-4-6-3-2-0-0-3-2534632003_11.mp4'

# Load categories from txt file
l_categories = utils.load_categories()
# Get index of the category from l_categories (from 0 to 338)
c_idx = [i for i in range(len(l_categories)) if l_categories[i] == c_name][0]

# Extract accuracies as (n_frames, 1) arrays
per_frame_accuracies = np.array(accuracies_per_category[c_name][f_name])

if c_idx in utils.topN_categories(per_frame_accuracies, N=N):
    print(f'In Top-{N}!')

################################################################################
# Check some random categories/files if in TopN
################################################################################
#%%
import random
random.seed(1)

N = 5
N_samples = 10
l_in_top = []
l_not_top = []

# Load categories from txt file
l_categories = utils.load_categories()

while True:
    if len(l_in_top) < N_samples or len(l_not_top) < N_samples:
        # draw random category
        rnd_category = l_categories[random.randint(0, len(l_categories)-1)]
        
        # Get index of the category from l_categories (from 0 to 338)
        c_idx = [i for i in range(len(l_categories))
                 if l_categories[i] == rnd_category][0]
        
        # draw random file from rnd_category
        l_files = list(accuracies_per_category[rnd_category].keys())
        
        if not l_files:
            continue
        else:
            rnd_file = l_files[random.randint(0, len(l_files)-1)]
            
            
            # Extract accuracies as (n_frames, 1) arrays
            per_frame_accuracies = np.array(accuracies_per_category[rnd_category][rnd_file])

            # check if in TopN
            if c_idx in utils.topN_categories(per_frame_accuracies, N=N):
                print(f'In Top-{N}!')
                l_in_top.append([rnd_category, rnd_file])
            else:
                print(f'NOT in Top-{N}')
                l_not_top.append([rnd_category, rnd_file])
    else:
        break

print('In TopN:')
print([x for x in l_in_top[:N_samples]])

print('\nNOT in TopN:')
print([x for x in l_not_top[:N_samples]])

################################################################################
# Extract inTopN category-accuracy pairs w/ best and worst frames
################################################################################
#%% With best and worst
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

for i in range(N_samples):
    c_name = l_in_top[i][0]
    f_name = l_in_top[i][1]
    
    # Load categories from txt file
    l_categories = utils.load_categories()
    # Get index of the category from l_categories (from 0 to 338)
    c_idx = [i for i in range(len(l_categories)) if l_categories[i] == c_name][0]

    labels, topN_, best, worst = utils.topN_per_file(accuracies_per_category, N=5,
                                                    category_name=c_name,
                                                    video_fname=f_name,
                                                    extract_best_worst=True)
        
    # Load video file using decord
    path_2_file = path_prefix / f'data/MIT_sampleVideos_RAW/{c_name}/{f_name}'
    vr = VideoReader(str(path_2_file))

    # Extract best and worst frames by their label
    best_idx  = best[0]
    worst_idx = worst[0]
    best_frame = vr.get_batch([best_idx])
    worst_frame = vr.get_batch([worst_idx])

    x=np.arange(topN_.shape[1])

    # Define figure
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(212)
    for i in range(topN_.shape[0]):
        if labels[i] != c_name:
            ax1.plot(x, topN_[i],
                    label=labels[i],
                    alpha=0.35)
        else:    
            ax1.plot(x, topN_[i],
                    label=labels[i],
                    linewidth=3)
    ax1.set_xlim(xmin=0, xmax=topN_.shape[1])
    ax1.set_title(f'Per-frame accuracies for {c_name} : {f_name}')
    ax1.set_xlabel('Frame nr.')
    ax1.set_ylabel('Prediction accuracy (softmax)')
    ax1.legend(title=f'Top-{len(topN_)} categories')

    ax2 = plt.subplot(221)
    #ax2.margins(2, 2)           
    ax2.imshow(worst_frame.asnumpy()[0])
    ax2.set_title(f'Worst frame ({worst_idx})')

    ax3 = plt.subplot(222)
    #ax3.margins(x=0, y=-0.25)  
    ax3.imshow(best_frame.asnumpy()[0])
    ax3.set_title(f'Best frame ({best_idx})')

    plt.tight_layout()
    #plt.savefig(str(path_prefix / f'plots/best&worst+top5/{c_name}_{f_name}.pdf'))
    plt.savefig(str(path_prefix / f'plots/best&worst+top5/inTop5/{c_name}_{f_name}.png'))
plt.show()

################################################################################
# Extract notTopN category-accuracy pairs w/ best and worst frames
################################################################################
#%% With best and worst
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

for i in range(N_samples):
    c_name = l_not_top[i][0]
    f_name = l_not_top[i][1]
    
    # Load categories from txt file
    l_categories = utils.load_categories()
    # Get index of the category from l_categories (from 0 to 338)
    c_idx = [i for i in range(len(l_categories)) if l_categories[i] == c_name][0]

    labels, topN_, best, worst = utils.topN_per_file(accuracies_per_category, N=5,
                                                    category_name=c_name,
                                                    video_fname=f_name,
                                                    extract_best_worst=True)
        
    # Load video file using decord
    path_2_file = path_prefix / f'data/MIT_sampleVideos_RAW/{c_name}/{f_name}'
    vr = VideoReader(str(path_2_file))

    # Extract best and worst frames by their label
    best_idx  = best[0]
    worst_idx = worst[0]
    best_frame = vr.get_batch([best_idx])
    worst_frame = vr.get_batch([worst_idx])

    x=np.arange(topN_.shape[1])

    # Define figure
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(212)
    for i in range(topN_.shape[0]):
        if labels[i] != c_name:
            ax1.plot(x, topN_[i],
                    label=labels[i],
                    alpha=0.35)
        else:    
            ax1.plot(x, topN_[i],
                    label=labels[i],
                    linewidth=3)
    ax1.set_xlim(xmin=0, xmax=topN_.shape[1])
    ax1.set_title(f'Per-frame accuracies for {c_name} : {f_name}')
    ax1.set_xlabel('Frame nr.')
    ax1.set_ylabel('Prediction accuracy (softmax)')
    ax1.legend(title=f'Top-{len(topN_)} categories')

    ax2 = plt.subplot(221)
    #ax2.margins(2, 2)           
    ax2.imshow(worst_frame.asnumpy()[0])
    ax2.set_title(f'Worst frame ({worst_idx})')

    ax3 = plt.subplot(222)
    #ax3.margins(x=0, y=-0.25)  
    ax3.imshow(best_frame.asnumpy()[0])
    ax3.set_title(f'Best frame ({best_idx})')

    plt.tight_layout()
    #plt.savefig(str(path_prefix / f'plots/best&worst+top5/{c_name}_{f_name}.pdf'))
    plt.savefig(str(path_prefix / f'plots/best&worst+top5/notTop5/{c_name}_{f_name}.png'))
plt.show()
# %%
