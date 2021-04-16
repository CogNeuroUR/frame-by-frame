# [15.04.21] OV
# Collects widths of given video set
################################################################################
# Imports
################################################################################
#%% Imports
# Image/video manipulation
import decord
decord.bridge.set_bridge('native')
from decord import VideoReader
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
import utils
importlib.reload(utils) # Reload If modified during runtime

#%%#############################################################################
# Collect list of files
################################################################################
#%% Define input paths
path_files = Path('../data/MIT_sampleVideos_RAW_final_25FPS/')

#%% Sweep through input gifs set
extension = 'mp4'

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
l_resolutions = []
# ==========================================================

l_cats = [] 
for i in range(len(l_files)):
  category, file_name = l_files[i]
  
  # Verbose 
  if i % 50 == 0: print(f'{i}/{len(l_files)}')

  # Load video file using decord
  path_2_file = path_files / category / file_name 
  
  # Check if file exists:
  if os.path.exists(path_2_file):
    vr = VideoReader(str(path_2_file))
    
    # Get sizes
    frame = vr.get_batch([0])
    
    # Append resolution per file (height, width)
    l_resolutions.append([category, file_name, frame.shape[1], frame.shape[2]]) 

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
res_df.to_csv('spreadsheets/MIT_sampleVideos_RAW_final_25FPS_resolutions.csv')
