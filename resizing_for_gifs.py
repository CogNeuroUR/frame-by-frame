# [21.01.21] OV
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
l_categories = utils.load_categories()
l_best = []
N = len(l_categories)
i = 0
path_to_dataset = path_prefix / 'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/'

# Sweep through dictionary and extract best frame idxs
for category in l_categories[:N]:
    #print(i, category)
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
            # Check if file exists:
            if os.path.exists(path_to_dataset / str(category) / str(file_name)):
                best, worst= utils.best_worst(accuracies_dict=accuracies_per_category,
                                            category_name=category,
                                            video_fname=file_name)
                l_best.append([category, file_name, best[0]])

# Order alphabetically the categories
if l_best:
    l_sorted_best = sorted(l_best)
    print('Indexes extracted and alphabetically sorted.')
else:
    print('Something went wrong! Check!')
################################################################################
#%% Resizing w/ ffmpeg (python subprocess) and create GIFs
################################################################################
width = 960
category = 'bicycling'
file_name = 'yt-5r5WH6nBey8_235.mp4'
path_2_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}').absolute())
out_file = str(Path(f'data/resizing_tests/w{width}_{category}_{file_name}').absolute())

subprocess.call(
    ['ffmpeg', '-i', path_2_file,  '-vf', f'scale={width}:-2', out_file])
print('file %s saved' % out_file)

#%% Test on a subset
N = 20
np.random.seed(0)
temp = np.arange(len(l_best))
np.random.shuffle(temp)
subset_idxs = temp[:N]
subset = np.array(l_best)[subset_idxs]

for category, file_name, best in subset:
  print(category, file_name)

#%% GIFs with decord + moviepy
# Parameters: ==============================================
start = time.time()
l_fps = []
T = 0.5 # time duration of the wanted segments (in seconds)
width = 960

out_folder = Path(f'data/resizing_tests/GIFs_original/').absolute()
# Check if exist, if not, make one
os.makedirs(out_folder, exist_ok=True)

save_all_three = False
i = 0
for category, file_name, best in subset:
    # Verbose 
    print(f'{i}/{len(subset)}')
    i+=1
    best = int(best)
    path_2_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}').absolute())
    output_file = str(out_folder / f'w{width}_{category}_{file_name}')
    # Load file
    vr = VideoReader(str(path_2_file))
    
    # Get FPS
    #l_fps.append(vr.get_avg_fps())
    #print(f'FPS: {vr.get_avg_fps()}')
    fps = vr.get_avg_fps()
    
    # Define the number of frames to be extracted
    N_frames = int(fps * T)
    
    # Define list with indices of frames ~ the position of the best frame
    l_best_begin = [i for i in range(best, best + N_frames)]
    l_best_mid = [i for i in range(best - N_frames//2, best + N_frames//2)]
    l_best_end = [i for i in range(best - N_frames, best)]
    
    # Empty arrays:
    frames_begin = np.array([])
    frames_mid = np.array([])
    frames_end = np.array([]) 
    
    # Extract frames w/ decord
    if l_best_begin[-1] < vr.__len__():
        frames_begin = vr.get_batch(l_best_begin).asnumpy()
    if l_best_mid[-1] < vr.__len__():
        frames_mid = vr.get_batch(l_best_mid).asnumpy()
    frames_end = vr.get_batch(l_best_end).asnumpy()
    
    print(len(l_best_begin), frames_begin.shape)
    print(len(l_best_mid), frames_mid.shape)
    print(len(l_best_end), frames_end.shape)
    
    # Define paths
    start_path = out_folder / f'w{width}_{category}_{file_name}_start.gif'
    mid_path = out_folder / f'w{width}_{category}_{file_name}_mid.gif'
    end_path = out_folder / f'w{width}_{category}_{file_name}_end.gif'

    # Save arrays to gifs:
    # IF "best-in-the-middle" exists, save it, otherwise, go to "...-begin", ...
    if save_all_three == False: # Save only "mid" OR "begin" OR "end"
        if frames_mid.size != 0:
            utils.gif(mid_path, frames_mid, int(fps), scale_width=width)
        elif frames_begin.size != 0:
            utils.gif(start_path, frames_begin, int(fps), scale_width=width)
        else:
            utils.gif(end_path, frames_end, int(fps), scale_width=width)
    else: # OR save all three (if not empty)
        if frames_begin.size != 0:
            utils.gif(start_path, frames_begin, int(fps), scale_width=width)
        if frames_mid.size != 0:
            utils.gif(mid_path, frames_mid, int(fps), scale_width=width)
        if frames_end.size != 0:
            utils.gif(end_path, frames_end, int(fps), scale_width=width)

stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.4f}s per file)')

#%% Having original-shape GIFs let's crop them to 4x3 and 1x1 (ffmpeg)
start = time.time()
# Parameters: ==============================================
width = 960 #mean_width
to_landscape = True # i.e. to 4x3
if to_landscape:
    out_folder = Path(f'data/resizing_tests/GIFs_4x3/').absolute()
else:
    out_folder = Path(f'data/resizing_tests/GIFs_1x1/').absolute()
# Check if exist, if not, make one
os.makedirs(out_folder, exist_ok=True)
# ==========================================================
input_path = Path(f'data/resizing_tests/GIFs_original/').absolute()
_, _, files_subset = next(os.walk(input_path))

i = 0
for file_name in files_subset:
    # Verbose 
    print(f'{i}/{len(subset)}')
    path_2_file = str(input_path / file_name)

    # Check if to_landscape, portrait or square:
    _, height, width, _ = VideoReader(str(path_2_file)).get_batch([0]).shape
    
    if to_landscape:
        output_file = str(out_folder / f'4x3_{file_name}')
        if width > height: # landscape
            shapes = 'ih/3*4:ih'
            scales = f'{width}:-2'
        elif width < height: # ffmpeg -i INPUT -filter:v "crop=iw:3/4*iw" out.gif
            shapes = 'iw:3/4*iw'
            scales = f'{width}:3/4*{width}'
        else:
            shapes = 'ih:ih*3/4'
            scales = f'{width}:-2'
        cmd = ['ffmpeg', '-i', path_2_file,  '-filter:v',
               f"crop={shapes}, scale={scales}", output_file]
        subprocess.call(cmd)
        #subprocess.call(
        #['ffmpeg', '-i', path_2_file,  '-filter:v', f"crop={shapes}", output_file])
    else:
        shapes = f'{width}:{width}'
        output_file = str(out_folder / f'1x1_{file_name}')
        # For landscape inputs, upscale them s.t. their height = $(width) 
        if width > height: # landscape
            subprocess.call(
            ['ffmpeg', '-i', path_2_file,  '-filter:v',
             f"scale=-2:{width}, crop={shapes}", output_file])
        else:
            subprocess.call(
                ['ffmpeg', '-i', path_2_file,  '-filter:v', f"crop={shapes}", output_file])
    
    subprocess.call(
    #['ffmpeg', '-i', path_2_file,  '-filter:v', f"scale={width}:-2, crop=ih/3*4:ih", output_file])
    ['ffmpeg', '-i', path_2_file,  '-filter:v', f"crop={shapes}", output_file])
    #print('file %s saved' % out_file)
    i += 1
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.3f}s per file)')

#%% Start resizing  
start = time.time()
# Parameters: ==============================================
width = 960 #mean_width
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
