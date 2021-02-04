# [13.01.21] OV
################################################################################
# Imports
################################################################################
#%% Imports
from __future__ import print_function
# Image/video manipulation
import cv2
# Path and sys related
import os
import sys
import importlib
import time
import subprocess
from pathlib import Path
# Data types & numerics
import numpy as np
import pandas as pd
# My utils
sys.path.insert(0, '..')
import utils

#%% # Reload If modified during runtime
importlib.reload(utils) 

################################################################################
#%% Sweep through videos
################################################################################
#%% Paths
from pathlib import Path

path_input = Path('data/MIT_sampleVideos_RAW_WORK_IN_PROGRESS').absolute()
path_output = Path('data/TEST_RESIZE&CROP').absolute()

if not os.path.exists(path_output):
  os.makedirs(path_output)

#%% Sweep through files in subfolders of path_input
l_videos = []
for path, subdirs, files in os.walk(path_input):
  for name in files:
    if name[-3:] == 'mp4':
      l_videos.append([path.split('/')[-1],   # category
                       name])                 # file name
    else:
      print('Ignored: ', name)

if l_videos:
  l_videos = sorted(l_videos)
print('Total nr. of MP4s: ', len(l_videos))

#%% ############################################################################
# Resizing of the videos
################################################################################
#%% Test on one video
import subprocess
width = 480
#category = 'bicycling'
#file_name = 'yt-5r5WH6nBey8_235.mp4'
category = 'burying'
file_name = 'yt-_6awwB9VXzo_13.mp4'
path_2_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}').absolute())
out_file = path_output / file_name

output = str(subprocess.check_output(
  ['ffprobe', '-v', 'quiet', '-print_format', 'default', '-show_format', '-show_streams', path_2_file]
  , stderr=subprocess.STDOUT)).split('\\n')
print(output[9][6:], output[10][7:])

print(subprocess.call(
  ['ffmpeg', '-i', path_2_file,  '-vcodec', 'libx264', '-vf', f'scale={width}:-2', '-y', out_file]
  ))

#%% Run a subset
import time
import cv2
import subprocess

start = time.time()
# Parameters: ==============================================
out_width = 480 #mean_width
j = 0
i = 0
# ==========================================================   
        
for category, file_name in l_videos[:100]:
    # Verbose
  print(f'{j}/{len(l_videos)}'); j+=1

  # Define paths
  path_input_file = str(path_input / category/ file_name)
  path_output_file = str(path_output / category / file_name)
  
  # Create output category directory if not present
  if not os.path.exists(path_output / category):
    os.mkdir(path_output / category)
  
  # Check the width & height
  vcap = cv2.VideoCapture(path_input_file)
  width  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  
  # Check the W to H ratio for a proper 4x3 cut
  if width / height > 4/3:
    shapes = 'ih*4/3:ih'
  else:
    shapes = 'iw:iw*3/4'
  scales = f'{out_width}:-2'
  
  # Define command
  cmd = ['ffmpeg', '-i', path_input_file, '-vcodec', 'libx264',
      '-filter:v', f"crop={shapes}, scale={scales}", '-y', path_output_file]
  
  # Run
  if subprocess.call(cmd) != 0:
    raise Exception(f'Error while resizing {category}/{file_name}!')
  
  i += 1
    
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/i:.3f}s per file)')

# %%
