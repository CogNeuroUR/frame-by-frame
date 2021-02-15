# [11.02.21] OV
# Script writting GIFs based on MIFs

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

#%% Load MIF csv
path_mifs = Path('data/MIT_sampleVideos_RAW_final_25FPS/mifs.csv')
mifs = pd.read_csv(path_mifs, usecols=['category', 'fname', 'mif_idx'])
print(mifs)

#%% Define GIF extraction function
from moviepy.editor import ImageSequenceClip
def gif(filename, array, fps):
  # ensure that the file has the .gif extension
  fname, _ = os.path.splitext(filename)
  filename = fname + '.gif'

  # copy into the color dimension if the images are black and white
  if array.ndim == 3:
      array = array[..., np.newaxis] * np.ones(3)

  # make the moviepy clip
  clip = ImageSequenceClip(list(array), fps=fps)
  clip.write_gif(filename, fps=fps, program='ffmpeg', verbose=False)
  #return clip
  
#%% ############################################################################
# Test on a single video
################################################################################
category = 'dancing'
file_name = 'yt--70wXQICQe8_35.mp4'
path_2_file = Path(f'data/GIFs/{file_name}')

print(mifs.loc[(mifs['category'] == category) & (mifs['fname'] == file_name)]['mif_idx'])

#%% Extract GIfs
#===============================================================================
#Parameters
save_all_three = True
#path_input = Path('data/GIFs/input.mp4')
path_output = Path('data/GIFs/').absolute()
T = 1 # in seconds
#===============================================================================
best = mifs.loc[(mifs['category'] == category) & (mifs['fname'] == file_name)]['mif_idx'].values[0]

# Load video file using decord
vr = VideoReader(str(path_2_file))

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
start_path = path_output / f'{file_name}_b.gif'
mid_path = path_output / f'{file_name}_m.gif'
end_path = path_output / f'{file_name}_e.gif'

#########################################

# Save arrays to gifs:
# IF "best-in-the-middle" exists, save it, otherwise, go to "...-begin", ...
if frames_begin.size != 0:
  clip = gif(start_path, frames_begin, int(fps))
if frames_mid.size != 0:
  clip = gif(mid_path, frames_mid, int(fps))
if frames_end.size != 0:
  clip = gif(end_path, frames_end, int(fps))
        
#%%
#%% ############################################################################
# Extract from list of videos
################################################################################
import time, os

start = time.time()
# Parameters: ==============================================
T = 1.0 # time duration of the wanted segments (in seconds) 
save_all_three = True
path_output = Path(f'data/GIFs/MIT_GIFs_25FPS_480x360p_{T}s/').absolute()
path_2_dataset = Path(f'data/MIT_sampleVideos_RAW_final_25FPS_480x360p/')

if not os.path.exists(path_output):
        os.mkdir(path_output)
# ==========================================================

i=0
for index, row in mifs.iterrows():
    i+=1
    if i < len(mifs) + 1000:  # "+1000" to be sure :-P
        category, file_name, best = row
        
        path_output_category = path_output / category
        path_output_file = path_output_category / file_name[:-4]
        
        # Create output category directory if not present
        if not os.path.exists(path_output_category):
            os.mkdir(path_output_category)
        # Create file directory if not present
        if not os.path.exists(path_output_file):
            os.mkdir(path_output_file)
        
        path_2_file = path_2_dataset / category / file_name
        # Load video file using decord
        vr = VideoReader(str(path_2_file))
        fps = vr.get_avg_fps()
            
        # Define the number of frames to be extracted
        N_frames = int(fps * T)
        
        # Define list with indices of frames ~ the position of the best frame
        l_best_begin = [i for i in range(best, best + N_frames)]
        l_best_mid = [i for i in range(best - N_frames//2, best + N_frames//2)]
        l_best_end = [i for i in range(best - N_frames, best)]

        # Empty arrays:
        frames_begin = None
        frames_mid = None
        frames_end = None

        # Extract frames w/ decord
        if l_best_begin[-1] < vr.__len__():
            frames_begin = vr.get_batch(l_best_begin).asnumpy()
        if l_best_mid[-1] < vr.__len__():
            frames_mid = vr.get_batch(l_best_mid).asnumpy()
        frames_end = vr.get_batch(l_best_end).asnumpy()

        # Define paths
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

# %%
