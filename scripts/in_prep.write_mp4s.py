
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

import cv2
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
path_mifs = Path('../input_data/MIT_sampleVideos_RAW_final_25FPS_480x360p/mifs.csv')
mifs = pd.read_csv(path_mifs, usecols=['category', 'fname', 'mif_idx'])
print(mifs)

####
# Possibly, a merge between initial & additional sets would be required first
####

#%% Load QQ-positioning, i.e. sheet containing the position relative to the MIF:
# b : beginning, m : middle, e : end
path_positioning = Path('../temp/20210719_AB_fname_+_QQ-positioning.csv')
positioning = pd.read_csv(path_positioning, names=['fname', 'mif_position'])
print(positioning)

#%% Remove NaNs (rows containing only category and not fname & mif_position)
positioning_no_nans = positioning.dropna()
print(positioning_no_nans)

#%% Sweep through 
temp = []
counter_notfound = 0

for index, row in positioning_no_nans.iterrows():
  # Try to find filename from positioning df in mifs df
  # IF not found, through "oops"
  try:
    category, fname, mif_idx = mifs.iloc[np.where(mifs['fname'] == row.tolist()[0] + '.mp4')[0][0]].tolist()
    #print(category, fname, mif_idx)
    
    # append to temp: category, fname, mif_idx, mif_position
    temp.append([category, fname, mif_idx, row.tolist()[1]])
  except IndexError: # 
    counter_notfound += 1
    #print(f'\tOops, {row.tolist()[0]}.mp4 not found!')
  
print('Files found: ', len(temp))
print('Files missing: ', counter_notfound)

# convert temp to pd.DataFrame
mifs_positioning = pd.DataFrame(temp,
                                columns=['category', 'fname', 'mif_idx', 'mif_position'])

print(mifs_positioning.head())

#%% ############################################################################
# Test on a single video
################################################################################
# Parameters
category = mifs_positioning.loc[0]['category']
file_name = mifs_positioning.loc[0]['fname']
mif_idx = mifs_positioning.loc[0]['mif_idx']
mif_position = mifs_positioning.loc[0]['mif_position']

T = 1 # output time in seconds

path_2_file = path_mifs.parent / category/ file_name
path_output = Path(f'../input_data/test_write_mp4s/{category}').absolute()
utils.check_mkdir(path_output)


#%%
cv2_vr = cv2.VideoCapture(str(path_2_file))
fps = cv2_vr.get(cv2.CAP_PROP_FPS)
width = cv2_vr.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cv2_vr.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(cv2_vr)
print(fps)
print(width)
print(height)

#%% 

#%% OPENCV alone: Reading selected frames with opencv (from gluoncv documentation)
start_time = time.time()

cv2_vr = cv2.VideoCapture(str(path_2_file))
fps = cv2_vr.get(cv2.CAP_PROP_FPS)
width = cv2_vr.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cv2_vr.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Define list with indices of frames ~ the position of the best frame
N_frames = int(fps * T)
frames_list = None
if mif_position == '_b':
  frames_list = [i for i in range(mif_idx, mif_idx + N_frames)]
if mif_position == '_m':
  frames_list = [i for i in range(mif_idx - N_frames//2, mif_idx + N_frames//2)]
if mif_position == '_e':
  frames_list = [i for i in range(mif_idx - N_frames, mif_idx)]

# Define codec
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# For a list of codecs, see:
# https://www.fourcc.org/codecs.php

# Create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(str(path_output / file_name),
                      fourcc,
                      int(fps),
                      (int(width), int(height)))

for frame_idx in frames_list:
  cv2_vr.set(1, frame_idx)
  _, frame = cv2_vr.read()
  
  #frame = cv2.resize(frame, (int(width), int(height)))
  
  # Write the frame into the file 'output.avi'
  out.write(frame) 

cv2_vr.release()
out.release()

end_time = time.time()
print('OpenCV alone takes %4.4f seconds.' % ((end_time - start_time)/10))


#%% DECORD + OPENCV: Write selected frames to file
"""
5 times faster, BUT it writes normalized frames, instead of originals

TO BE SOLVED!
"""
start_time = time.time()
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
vr = VideoReader(str(path_2_file))

fps = vr.get_avg_fps()
# Define the number of frames to be extracted
N_frames = int(fps * T)

# Create empty array for collecting frames & Extract frames w/ decord
frames = np.array([])
frames = vr.get_batch(frames_list).asnumpy()
print(frames.shape)

out = cv2.VideoWriter(str(path_output / file_name),
                      cv2.VideoWriter_fourcc(*'MP4V'),
                      int(fps),
                      (int(width), int(height)))


for i in range(frames.shape[0]):
    out.write(frames[i])
out.release()

end_time = time.time()
print('Decord + OPENCV takes %4.4f seconds.' % ((end_time - start_time)/10))


#%% ############################################################################
# Extract MP4s from list of videos
################################################################################

"""
# TODO
# Identical to procedure to 06.write_gifs.py
"""