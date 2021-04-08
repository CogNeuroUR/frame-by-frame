# [08.04.21] OV
# Script for writting GIFs based on MIFs

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

#%% Load MIF csv
path_mifs = Path('../data/MIT_additionalVideos_25FPS_480x360p/mifs.csv')
mifs = pd.read_csv(path_mifs, usecols=['category', 'fname', 'mif_idx'])
print(mifs)

#%% Define GIF extraction function
from moviepy.editor import ImageSequenceClip
from moviepy.video.fx import blackwhite

def gif(filename, array, fps, rewrite=False, bw=False):
  """
  Writes GIFs, given array of frames, using moviepy functionality
  
  Parameters
  ----------
  filename : str or pathlib.Path
    Path to output GIF file
  array : numpy.ndarray
    Frames array (n_frames, height, width, n_channels)
  fps : int
    Output FPS
  rewrite : bool
    True if to rewrite output file (Default: False)
  bw : bool
    True if Black&White (Default: False)
    
  Returns
  -------
  None
  """
  
    # Check if files already exists:
  if os.path.exists(filename):
    if rewrite == True:
      pass
    else:
      raise Exception('File already exists. Exiting...')
  
  # copy into the color dimension if the images are black and white
  if array.ndim == 3:
      array = array[..., np.newaxis] * np.ones(3)

  # make the moviepy clip
  clip = ImageSequenceClip(list(array), fps=fps)
  if bw == True:
      clip = blackwhite.blackwhite(clip)
  clip.write_gif(filename, fps=fps, program='ffmpeg', verbose=False)

#%% ############################################################################
# Test on a single video
################################################################################
# Parameters
category = 'arresting'
file_name = 'yt-aAVfUYxx12g_18.mp4'
path_2_file = Path(f'../data/single_video_25FPS_480x360p/{category}/{file_name}')

T = 1 # in seconds
path_output = Path('../data/single_video_25FPS_480x360p_1s/').absolute()
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

# Empty arrays for collecting frames:
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
        gif(start_path, frames_begin, int(fps), rewrite=True)
if frames_mid is not None:
        gif(mid_path, frames_mid, int(fps), rewrite=True)
if frames_end is not None:
        gif(end_path, frames_end, int(fps), rewrite=True)

#%% ############################################################################
# Extract GIFs from list of videos
################################################################################
# Check time for elapsed
start = time.time()

# Parameters: ==============================================
T = 1.0 # time duration of the wanted segments (in seconds)
path_output = Path(f'data/GIFs/MIT_additionalGIFs_25FPS_480x360p_{T}s/').absolute()
path_2_dataset = Path(f'data/MIT_additionalVideos_25FPS_480x360p/')

utils.check_mkdir(path_output)

# ==========================================================
# Iterate over rows of mif dataframe 
i=0
for index, row in mifs.iterrows():
  i+=1
  if i < len(mifs) + 1000:  # "+1000" to be sure :-P
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
    if l_best_begin[-1] < vr.__len__():
        frames_begin = vr.get_batch(l_best_begin).asnumpy()
    if l_best_mid[-1] < vr.__len__():
        frames_mid = vr.get_batch(l_best_mid).asnumpy()
    frames_end = vr.get_batch(l_best_end).asnumpy()

    # Define output paths per MIF position
    start_path = path_output_file / f'{file_name[:-4]}_b.gif'
    mid_path = path_output_file / f'{file_name[:-4]}_m.gif'
    end_path = path_output_file / f'{file_name[:-4]}_e.gif'

    # Save arrays to gifs:
    # IF "best-in-the-middle" exists, save it, otherwise, go to "...-begin", ...
    if frames_begin is not None:
            gif(start_path, frames_begin, int(fps))
    if frames_mid is not None:
            gif(mid_path, frames_mid, int(fps))
    if frames_end is not None:
            gif(end_path, frames_end, int(fps))
  else:
    break

stop = time.time()
duration = stop-start
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
N = 20
path_dataset = Path('data/MIT_sampleVideos_RAW_final_25FPS_480x360p/')
path_output = Path('data/GIFs/BW_GIFs')
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
    start_path = path_output_category / f'{file_name[:-4]}_b.gif'
    mid_path = path_output_category / f'{file_name[:-4]}_m.gif'
    end_path = path_output_category / f'{file_name[:-4]}_e.gif'

    #########################################

    # Save arrays to gifs:
    # IF "best-in-the-middle" exists, save it, otherwise, go to "...-begin", ...
    if frames_begin.size != 0:
        gif(start_path, frames_begin, int(fps), bw=True, rewrite=True)
    if frames_mid.size != 0:
        gif(mid_path, frames_mid, int(fps), bw=True, rewrite=True)
    if frames_end.size != 0:
        gif(end_path, frames_end, int(fps), bw=True, rewrite=True)
# %%
