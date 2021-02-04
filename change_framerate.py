# [02.02.21] OV
# Script to change framerate of videos given a parent folder
#%%#############################################################################
# Imports
################################################################################
import time
import ffmpeg
from pathlib import Path
import pickle
import os
import sys, importlib
sys.path.insert(0, '..')
import utils

#%% Reload If modified during runtime
importlib.reload(utils)

################################################################################
#%% Sweep through videos
################################################################################
#%% Paths
path_input = Path('data/MIT_sampleVideos_RAW_WORK_IN_PROGRESS').absolute()
path_output = Path('data/TEST_FPS').absolute()

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

#%% Sweep through files and change framerate to given i_fps
import subprocess

i_fps = 25
keep_audio = True

start = time.time()

j = 0
i = 0
for category, file_name in l_videos[:1]:
  # Verbose
  print(f'{j}/{len(l_videos)}'); j+=1

  path_input_file = str(path_input / category/ file_name)
  path_output_file = str(path_output / category / file_name)
  
  # Create output category directory if not present
  if not os.path.exists(path_output / category):
    os.mkdir(path_output / category)
  
  # Remove file in output dir if present
  if os.path.exists(path_output_file):
    os.remove(path_output_file)
  
  if keep_audio == False: # Do not keep audio
    l_cmd = ['ffmpeg', '-i', path_input_file, '-r', str(i_fps), '-c:v', 'copy', '-an', '-y', path_output_file]
    
    out = subprocess.call(l_cmd)
    if out != 0:
      print('Error at ', [category, file_name])
  else: # Keep audio
    l_cmd = ['ffmpeg', '-i', path_input_file, '-r', str(i_fps), '-c:v', 'copy', '-y', path_output_file]
    
    out = subprocess.call(l_cmd)
    if out != 0:
      print('Error at ', [category, file_name])
    
  # Increment
  i+=1
    
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/i:.2f}s per file)')


# %%
subprocess.call(['ffmpeg', '-i', path_input_file, '-r', str(i_fps), '-c:v', 'copy', '-y', path_output_file])

# %%
