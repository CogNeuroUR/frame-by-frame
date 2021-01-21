# [13.01.21] OV
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
from sklearn.cluster import AgglomerativeClustering
# Plots
import seaborn as sns
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
path_prefix = Path().parent.absolute()
dict_path = path_prefix / 'saved/full/accuracies_per_category_full_mitv1.pkl'
# Load from file
f = open(dict_path, 'rb')
accuracies_per_category = pickle.load(f)
# Print the categories
#print(len(list(accuracies_per_category.keys())))

#%% ############################################################################
# Extract best frame index from all available videos
################################################################################
# Load categories from txt file
import time

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

#%% Order alphabetically the categories
l_sorted_best = sorted(l_best)
print(l_sorted_best[:10])

#%% ############################################################################
# Collect widths and heights
################################################################################
start = time.time()
# Parameters: ==============================================
l_resolutions = []
# ==========================================================

i = 0
l_cats = [] 
for category, file_name, best in l_sorted_best:
    # Verbose 
    print(f'{i}/{len(l_sorted_best)}')
  
    
    # Load video file using decord
    path_2_file = Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}')
    
    # Check if file exists:
    if os.path.exists(path_2_file):
      vr = VideoReader(str(path_2_file))
      
      # Get sizes
      frame = vr.get_batch([0])
      
      l_cats.append(category)
      if i == 0:
        l_resolutions.append([category, ''])
        l_resolutions.append([file_name, frame.shape[1], frame.shape[2]])
      else:
        if l_cats[i-1] == category:
          l_resolutions.append([file_name, frame.shape[1], frame.shape[2]])
        else:
          l_resolutions.append([category, ''])
          l_resolutions.append([file_name, frame.shape[1], frame.shape[2]])
          
    
      i+=1
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.2f}s per file)')
# %%
print(l_resolutions[:10])
# %%
res_df = pd.DataFrame(l_resolutions, columns=['Categories/File names', 'Height', 'Width'])
print(res_df)
#%%
res_df.to_csv('outputs/RAW_DOWNSIZING_IN_PROGRESS_resolutions_sorted.csv')

#%% ############################################################################
# Resizing of the videos
################################################################################
#%% Load resolutions from csv
res_df = pd.read_csv('outputs/RAW_DOWNSIZING_IN_PROGRESS_resolutions_sorted.csv')
print(res_df)

#%% Option #1: Resize to average width (+ keep aspect ratio)
mean_width = int(res_df['Width'].mean())

#%% Function for resizing
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

################################################################################
#%% With ffmpeg (python subprocess)
################################################################################
width = mean_width
category = 'bicycling'
file_name = 'yt-5r5WH6nBey8_235.mp4'
path_2_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}').absolute())
out_file = str(Path(f'data/resizing_tests/w{width}_{category}_{file_name}').absolute())

subprocess.call(
    ['ffmpeg', '-i', path_2_file,  '-vf', f'scale={width}:-2', out_file])
print('file %s saved' % out_file)

#%% Test on a subset
N = 20
temp = np.arange(len(l_best))
np.random.shuffle(temp)
subset_idxs = temp[:N]
subset = np.array(l_best)[subset_idxs]

for category, file_name, best in subset:
  print(category, file_name)

#%% Start resizing  
start = time.time()
# Parameters: ==============================================
width = 1280 #mean_width
out_folder = Path(f'data/resizing_tests/comparison/').absolute()
# ==========================================================

i = 0
l_cats = [] 
for category, file_name, best in subset:
    # Verbose 
    print(f'{i}/{len(subset)}')
  
    path_2_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}').absolute())
    output_file = str(out_folder / f'w{width}_{category}_{file_name}')
    subprocess.call(
    ['ffmpeg', '-i', path_2_file,  '-vf', f'scale={width}:-2', output_file])
    #print('file %s saved' % out_file)
    i += 1
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.3f}s per file)')

#%% Copy the subset for comparison
import shutil
i=0
for category, file_name, best in subset:
    # Verbose 
    print(f'{i}/{len(subset)}')
  
    # Load video file using decord
    path_2_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}').absolute())    
    # Get width
    if os.path.exists(path_2_file):
        vr = VideoReader(str(path_2_file))
        original_width = vr.get_batch([0]).shape[1]
        
        output_file = str(out_folder / f'ow{original_width}_{category}_{file_name}')
        
        shutil.copyfile(src=path_2_file,
                        dst=output_file)
    i += 1

#%% ############################################################################
# Search for (black) border containing videos
################################################################################
#%% Sweep through videos

start = time.time()
# Parameters: ==============================================
l_borders = []
# ==========================================================

def has_border(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    #diff = ImageChops.add(diff, diff, 2.0, -100)
    diff = ImageChops.add(diff, diff)      
    bbox = diff.getbbox()
    return bbox != (0,0,im.size[0],im.size[1])

i = 0
l_cats = [] 
for category, file_name, best in l_sorted_best:
    # Verbose 
    print(f'{i}/{len(l_sorted_best)}')
  
    
    # Load video file using decord
    path_2_file = Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}')
    
    # Check if file exists:
    if os.path.exists(path_2_file):
      #vr = VideoReader(str(path_2_file))
      cap = cv2.VideoCapture(str(path_2_file))
      
      # Get sizes
      #frame = vr.get_batch([0]).asnumpy()[0]
      _, frame = cap.read()
      """
      #img = cv2.imread('sofwin.png')
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

      contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      cnt = contours[0]
      x,y,w,h = cv2.boundingRect(cnt)
      """
      #print(f'{category}/{file_name}')
      #print(frame.shape)
      #print(has_border(Image.fromarray(frame)))
      
      """
      plt.imshow(frame)
      plt.show()
      """
      if has_border(Image.fromarray(frame)):
        l_borders.append([category, file_name])
          
    
      i+=1
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.2f}s per file)')

# %%
print(l_borders)

# %%
bordered_df = pd.DataFrame(l_borders, columns=['Categories', 'File names'])
print(bordered_df)
#%%
bordered_df.to_csv('outputs/RAW_DOWNSIZING_IN_PROGRESS_bordered_videos.csv')

# %%
