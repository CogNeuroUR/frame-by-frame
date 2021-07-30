# [08.04.21] OV
# Script for writting GIFs based on MIF indices

################################################################################
# Imports
################################################################################
#%% Imports
from __future__ import print_function
# Decord related
import decord # AB: VSC marks some import commands as problematic ("Unable to import 'decord'pylint(import-error)")
# Set decord output type to native, i.e. numpy.ndarray
decord.bridge.set_bridge('native')  # Alternatives: mxnet, torch and tensorflow
from decord import VideoReader
# Path and sys related
import os
from pathlib import Path
# Data types & numerics
import numpy as np
import pandas as pd
import time, os

#%% Import custom utils
import sys, importlib
sys.path.insert(0, '..')
import utils
importlib.reload(utils) # Reload If modified during runtime

#%% Load MIF csv created during the MIF extraction step, i.e. 01.mif_extraction*.ipynb
path_mifs = Path('../input_data/MIT_additionalVideos_25FPS_480x360p/mifs.csv')
mifs = pd.read_csv(path_mifs, usecols=['category', 'fname', 'mif_idx'])
print(mifs)

#%% ############################################################################
# Test on a single video
################################################################################
# AB: Same as in '05.classification_visualization.py':
# a bit more clear that this is an example run

# Parameters
category = 'arresting'
file_name = 'yt-aAVfUYxx12g_18.mp4'
path_2_file = Path(f'../input_data/single_video_25FPS_480x360p/{category}/{file_name}')
# AB: change path if necessary

T = 1 # in seconds
path_output = Path(f'../input_data/GIFs/single_video_25FPS_480x360p_1.0s/{category}').absolute()
utils.check_mkdir(path_output)

#===============================================================================
# Get MIF index
best = mifs.loc[(mifs['category'] == category) & (mifs['fname'] == file_name)]['mif_idx'].values[0]

# Load video file using decord
vr = VideoReader(str(path_2_file))
# Get FPS
fps = vr.get_avg_fps()
# Define the number of frames to be extracted
N_frames = int(fps * T)

# Define list with indices of frames ~ the position of the best frame
l_best_begin = [i for i in range(best, best + N_frames)]
l_best_mid = [i for i in range(best - N_frames//2, best + N_frames//2)]
l_best_end = [i for i in range(best - N_frames, best)]

# Create empty arrays for collecting frames:
frames_begin = np.array([])
frames_mid = np.array([])
frames_end = np.array([]) 

# Extract frames w/ decord
if l_best_begin[-1] < vr.__len__(): # check if last proposed frame idx exists 
    frames_begin = vr.get_batch(l_best_begin).asnumpy()
if l_best_mid[-1] < vr.__len__(): # check if last proposed frame idx exists 
    frames_mid = vr.get_batch(l_best_mid).asnumpy()
frames_end = vr.get_batch(l_best_end).asnumpy()

# Verbose
print(len(l_best_begin), frames_begin.shape)
print(len(l_best_mid), frames_mid.shape)
print(len(l_best_end), frames_end.shape)

# Append suffix depending on MIF position (beginning, middle or end)
start_path = path_output / f'{file_name[:-4]}_b.gif'
mid_path = path_output / f'{file_name[:-4]}_m.gif'
end_path = path_output / f'{file_name[:-4]}_e.gif'

#########################################

# Save arrays to gifs:
# IF "best-in-the-middle" exists, save it, otherwise, go to "...-begin", ...
# Save arrays to gifs:
        # IF "best-in-the-middle" exists, save it, otherwise, go to "...-begin", ...
if frames_begin is not None:
        utils.gif(start_path, frames_begin, int(fps), rewrite=True)
if frames_mid is not None:
        utils.gif(mid_path, frames_mid, int(fps), rewrite=True)
if frames_end is not None:
        utils.gif(end_path, frames_end, int(fps), rewrite=True)

#%% ############################################################################
# Extract GIFs from list of videos
################################################################################
# Check time for elapsed
start = time.time()

# AB: is this then extracted for ALL videos of the given path
# I think yes --> then better declare it here explicitely to render code more transperent please

# Parameters: ==============================================
T = 1.0 # time duration of the wanted segments (in seconds)
# This is an important parameter
# AB:  Change these if necessary
path_output = Path(f'input_data/GIFs/MIT_additionalGIFs_25FPS_480x360p_{T}s/').absolute()
path_2_dataset = Path(f'input_data/MIT_additionalVideos_25FPS_480x360p/{category}/')

utils.check_mkdir(path_output)

# ==========================================================
# Iterate over rows of mif dataframe 
i=0
for index, row in mifs.iterrows(): # AB: Does this loop iterate e.g. over all our MIT-vodeos? if yes please satet her additionaly
  i+=1
  if i < len(mifs) + 1000:  # "+1000" to be sure :-P
    # AB: What is this extension for?
    # Extract category, filename and MIF idx
    category, file_name, best = row
    
    # Define output paths
    path_output_category = path_output / category
    path_output_file = path_output_category / file_name[:-4]
    
    # Create output category and file directories if not present
    utils.check_mkdir(path_output_category)
    utils.check_mkdir(path_output_file)
    
    # Path to input file
    path_2_file = path_2_dataset / category / file_name
    
    # Load video file using decord
    vr = VideoReader(str(path_2_file))
    # Get FPS
    fps = vr.get_avg_fps()
    # Define the number of frames to be extracted
    N_frames = int(fps * T)
    
    # Define list with indices of frames ~ the position of the best frame
    l_best_begin = [i for i in range(best, best + N_frames)]
    l_best_mid = [i for i in range(best - N_frames//2, best + N_frames//2)]
    l_best_end = [i for i in range(best - N_frames, best)]

    # Empty arrays for collecting frames:
    frames_begin = None
    frames_mid = None
    frames_end = None

    # Extract frames w/ decord
    # (also check if last proposed frame idx exists )
    if l_best_begin[-1] < vr.__len__(): # vr := video reader (decord) (?)
        frames_begin = vr.get_batch(l_best_begin).asnumpy()
    if l_best_mid[-1] < vr.__len__():
        frames_mid = vr.get_batch(l_best_mid).asnumpy()
    frames_end = vr.get_batch(l_best_end).asnumpy()

    # Define output paths per MIF position
    start_path = path_output_file / f'{file_name[:-4]}_b.gif'
    mid_path = path_output_file / f'{file_name[:-4]}_m.gif'
    end_path = path_output_file / f'{file_name[:-4]}_e.gif'

    # AB: Does the batch saving of the gifs with the names specified above happen HERE?
    # if yes make more clear plese

    # Save arrays to gifs:
    # IF "best-in-the-middle" exists, save it, otherwise, go to "...-begin", ...
    if frames_begin is not None:
            utils.gif(start_path, frames_begin, int(fps))
    if frames_mid is not None:
            utils.gif(mid_path, frames_mid, int(fps))
    if frames_end is not None:
            utils.gif(end_path, frames_end, int(fps))
  else:
    break

stop = time.time()
duration = stop-start
# runtime feedback to user
print(f'\nTime spent: {duration:.2f}s (~ {duration/i:.3f}s per file)')

#%% Sweep through files in subfolders of path_input
l_videos = []
for path, subdirs, files in os.walk(path_output):
  for name in files:
    if name[-3:] == 'gif':
      l_videos.append([path.split('/')[-1],   # category
                       name])                 # file name
    else:
      print('Ignored: ', name)

if l_videos:
  l_videos = sorted(l_videos)
print('Total nr. of GIFs: ', len(l_videos))

#%% ############################################################################
# Test on a subset
################################################################################
#%%
# declare size of subset to test upon
N = 20
# change paths if necessary
path_dataset = Path('input_data/MIT_sampleVideos_RAW_final_25FPS_480x360p/')
path_output = Path('input_data/GIFs/BW_GIFs')
subset = mifs.sample(n=N, random_state=2)

for index, row in subset.iterrows():
    category, file_name, best = row
    
    path_output_category = path_output / category
    
    # Create output category directory if not present
    if not os.path.exists(path_output_category):
        os.mkdir(path_output_category)
            
    vr = VideoReader(str(path_dataset / category / file_name))
    
    best = mifs.loc[(mifs['category'] == category) & (mifs['fname'] == file_name)]['mif_idx'].values[0]
    
    mif = vr[best].asnumpy()
    fps = vr.get_avg_fps()
        
    # Define the number of frames to be extracted
    N_frames = int(fps * T)

    """
    TODO: Make sure to prevent negative indices!!! [OV, 30.07.21]
    """
    # Define list with indices of frames ~ the position of the best frame
    l_best_begin = [i for i in range(best, best + N_frames)]                # MIF := beginning fram
    l_best_mid = [i for i in range(best - N_frames//2, best + N_frames//2)] # MIF := middle frame
    l_best_end = [i for i in range(best - N_frames, best)]                  # MIF := end frame

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
    start_path = path_output_category / f'{file_name[:-4]}_b.gif'
    mid_path = path_output_category / f'{file_name[:-4]}_m.gif'
    end_path = path_output_category / f'{file_name[:-4]}_e.gif'

    #########################################

    # Save arrays to gifs:
    # IF "best-in-the-middle" exists, save it, otherwise, go to "...-begin", ...
    if frames_begin.size != 0:
        utils.start_path, frames_begin, int(fps), bw=True, rewrite=True)
    if frames_mid.size != 0:
        utils.gif(mid_path, frames_mid, int(fps), bw=True, rewrite=True)
    if frames_end.size != 0:
        utils.gif(end_path, frames_end, int(fps), bw=True, rewrite=True)
# %%
