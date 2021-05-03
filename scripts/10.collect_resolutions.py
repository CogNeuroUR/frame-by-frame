# [15.04.21] OV
# Collects the resolution (width) of videos in the video set
################################################################################
# Imports
################################################################################
#%% Imports
# Image/video manipulation
import decord # AB: unable to import; needs installation first? how? (please comment)
decord.bridge.set_bridge('native')
from decord import VideoReader # AB:  same see above
import cv2
# Path and sys related
import os
import sys
import time
from pathlib import Path
# Data types & numerics
import numpy as np
import pandas as pd
# Plots
import seaborn as sns
import matplotlib.pyplot as plt

#%% Import custom utils
import sys, importlib
sys.path.insert(0, '..')
import utils # AB: Here I'm not able to improt utils (do you know why? (is there an reuirement?)
importlib.reload(utils) # Reload If modified during runtime

#%%#############################################################################
# Collect list of files
################################################################################
#%% Define input paths
# change path if necessary
path_files = Path('../input_data/MIT_sampleVideos_RAW_final_25FPS/')

#%% Sweep through input gifs set
extension = 'mp4' # a change of format would be possible here (?)

l_files = []
l_missing = []

for path, subdirs, files in os.walk(path_files):
  for name in files:
    if name[-3:] == extension:
      l_files.append([path.split('/')[-1],   # category
                       name])                # file name
    else:
      print('Ignored: ', name)
      
    if len(files) < 3:
      l_missing.append([path.split('/')[-1], len(files)])

if l_missing:
  l_missing.pop(0)

if l_files:
  l_files = sorted(l_files)
print('Total nr. of files: ', len(l_files))

#%% ############################################################################
# Collect widths and heights
################################################################################
start = time.time()
# Parameters: ==============================================
l_resolutions = [] #  initialize empty resolution list
# ==========================================================

l_cats = [] 
for i in range(len(l_files)):
  category, file_name = l_files[i]
  
  # Verbose 
  # feedback to user every 50 cases (?)
  if i % 50 == 0: print(f'{i}/{len(l_files)}')

  # Load video file using decord
  path_2_file = path_files / category / file_name 
  
  # Check if file exists:
  if os.path.exists(path_2_file):
    vr = VideoReader(str(path_2_file))
    
    # Get sizes
    frame = vr.get_batch([0]) # AB: this only loads the video in the loop (?)
    
    # Append resolution per file (height, width)
    l_resolutions.append([category, file_name, frame.shape[1], frame.shape[2]])
    # AB: here the resolution is extracted via shape (?)

stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.2f}s per file)')

# %% Examine
print(l_resolutions[:10])

# %% Convert to DataFrame
res_df = pd.DataFrame(l_resolutions,
                      columns=['Categories', 'File names', 'Height', 'Width'])
print(res_df)

#%% Write to csv
# AB: What is then displayed in this csv file? (compare df above)
# Categories | File names | Height | Width
res_df.to_csv('spreadsheets/MIT_sampleVideos_RAW_final_25FPS_resolutions.csv')
