# [30.03.21] OV
# Script to change framerate of videos given a parent folder using ffmpeg
#%%#############################################################################
# (on MacOS) Switch to stock python which has access to homebrew installation 
# of ffmpeg -> let's us encode back into h264 (no visual loss)
################################################################################
#%%#############################################################################
# Imports
################################################################################
import time
from pathlib import Path
import os
import subprocess

#%%#############################################################################
# Sweep through videos
################################################################################
#%% Paths
# AB: adjust paths (location to RAW video files + ourput dir)
path_input = Path('../input_data/MIT_sampleVideos_RAW').absolute()
#path_output = Path('input_data/MIT_sampleVideos_RAW_25FPS').absolute()
path_output = Path('../input_data/TEST_FPS').absolute()

# Check if output path exists, if not, make one
if not os.path.exists(path_output):
  print(f'Ouptput path not found. Creating one at {str(path_output.absolute())}')
  os.makedirs(path_output)

#%% Sweep through files in subfolders of path_input to collect fnames
l_videos = []
for path, subdirs, files in os.walk(path_input):
  for name in files:
    if name[-3:] == 'mp4': # "x[-3:]" takes last 3 elem.s of x
      l_videos.append([path.split('/')[-1],   # category
                       name])                 # file name
    else:
      print('Ignored: ', name)

if l_videos:
  l_videos = sorted(l_videos)
print('Total nr. of MP4s: ', len(l_videos)) # AB: user feedback

#%% Sweep through collected fnames and change framerate to given i_fps
# Parameters
i_fps = 25          # output fps
keep_audio = True   # whether to keep audio channel in output files

# For time measurements
start = time.time()

j = 0 # for verbose & time measurements
for category, file_name in l_videos[:10]:
  # Verbose
  print(f'{j}/{len(l_videos)}'); j+=1 # user feedback / verbose

  # Create output category directory if not present
  if not os.path.exists(path_output / category):
    os.mkdir(path_output / category)
    
  # Define input- & output-paths per file
  path_input_file = str(path_input / category/ file_name)
  path_output_file = str(path_output / category / file_name)
  
  # Remove file in output dir if present
  if os.path.exists(path_output_file):
    os.remove(path_output_file)
  
  # Change framerate
  if keep_audio == False: # Do not keep audio
    # Define command for subprocess
    l_cmd = ['ffmpeg', '-i', path_input_file, '-c:v', 'libx264', '-r', str(i_fps), '-an', '-y', path_output_file]
    # for details on commands: http://ffmpeg.org/ffmpeg.html
    # Run command
    out = subprocess.call(l_cmd)
    # Check for errors
    if out != 0:
      print('Error at ', [category, file_name])
  
  else: # Keep audio
    # Define command for subprocess
    l_cmd = ['ffmpeg', '-i', path_input_file, '-c:v', 'libx264', '-r', str(i_fps), '-y', path_output_file]
    # Run command
    out = subprocess.call(l_cmd)
    # Check for errors
    if out != 0:
      print('Error at ', [category, file_name])
   
# Compute time duration  
stop = time.time()
duration = stop-start
print(f'\nTime elapsed: {duration:.2f}s (~ {duration/j:.2f}s per file)')