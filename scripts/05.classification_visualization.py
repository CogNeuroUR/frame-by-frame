# [01.04.21] OV
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
# Path and sys related
import pickle
from pathlib import Path
# Data types
import numpy as np
# Plots
import matplotlib
#%matplotlib qt
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
path_prefix = Path('..')
# AB: Obtained in which step? (adds to understandability)
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
c_name = 'chopping' # because it's loadead in repo (AB: GitHub repo?)
f_name =  'flickr-3-4-6-3-2-0-0-3-2534632003_11.mp4'

# AB: I understand the code in that way, that one would have to adjsut the c_name/f_name
# if one wnated a figure displayed a different action?
# Back then you provided me some ~ 20 figures, I guess all made individualyl with this scirpt?

# This could be speciefied a bit more here to be more clear, plese

# Load categories from txt file
# AB: change path if necessary
l_categories = utils.load_categories('../labels/category_momentsv1.txt')

# Get index of the category from l_categories (from 0 to 338)
c_idx = [i for i in range(len(l_categories)) if l_categories[i] == c_name][0]

#%%#############################################################################
# Extract TopN category-accuracy pairs (clean)
################################################################################
# AB: more info considering the utils (also in readme / doc would help to understand this step)
labels, topN_ = utils.topN_per_file(accuracies_per_category,
                                    l_categories=l_categories,
                                    N=5,
                                    category_name=c_name,
                                    video_fname=f_name)
print(len(topN_))

################################################################################
# Plots (clean)
################################################################################
#%% Plot worst vs best frame PLUS accuracy evolution
plt.figure(figsize=(20, 10))
ax1 = plt.subplot(212)

x=np.arange(topN_.shape[1])

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
plt.tight_layout()
plt.show()

################################################################################
# Extract TopN category-accuracy pairs w/ best and worst frames
################################################################################
#%% With best and worst
c_name = 'cutting'
f_name =  'yt-UgO4jE-puiE_67.mp4'

# Get index of the category from l_categories (from 0 to 338)
c_idx = [i for i in range(len(l_categories)) if l_categories[i] == c_name][0]

labels, topN_, best, worst = utils.topN_per_file(accuracies_per_category, N=5,
                                                 l_categories=l_categories,
                                                 category_name=c_name,
                                                 video_fname=f_name,
                                                 extract_best_worst=True)

################################################################################
# Plots (+ best & worth)
################################################################################
#%% Plot with best and worst
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
#plt.savefig(str(path_prefix / f'plots/best&worst+top5/{c_name}_{f_name}.png'))
plt.show()

# %%
