# [11.02.21] OV
# Script writting GIFs based on MIFs using ffmpeg

#*******************************************************************************
# Note that for this script, (on macOS) the homebrew installation of ffmpeg & ffprobe
# was used!
#*******************************************************************************

# AB: - INFO - 
# SoSci Survey 8https://www.soscisurvey.de/; 2021-04-20), has an upload quota
# for GIFs which is different than for MP4s (640 kB vs. 64 MB). 
# To use the larger upload cap, a transformation from GIFs 'bacl' to MP4s seems
# plausible, to implement the stimuli for the experiment properly in SSS.

################################################################################
# Imports
################################################################################
#%% Imports
from pathlib import Path
import time
import cv2
import subprocess
import os

#%% Sweep through videos
# adjust these paths if necessary
path_input = Path('../input_data/GIFs/MIT_GIFs_25FPS_480x360p_N=834_renamed').absolute()
path_output = Path('../input_data/MP4s/MIT_MP4s_25FPS_480x360p_N=834_renamed').absolute()

if not os.path.exists(path_output):
  os.makedirs(path_output)

#%% Sweep through files in subfolders of path_input
l_videos = []
for path, subdirs, files in os.walk(path_input):
  for name in files:
    # Check if files have .mp4 extension
    if name[-3:] == 'gif':
      l_videos.append([path.split('/')[-1],   # category
                       name])                 # file name
    # Report non-mp4 files
    else:
      print('Ignored: ', name)

if l_videos:
  l_videos = sorted(l_videos)
print('Total nr. of GIFs: ', len(l_videos))

#%% ############################################################################
# Conversion: GIF -> MP4 using ffmpeg
################################################################################
# ffmpeg is a very fast video and audio converter that can also grab from a live
# audio/video source.
# It can also convert between arbitrary sample rates and resize video on the fly
# with a high quality polyphase filter.
# (Sourse: https://ffmpeg.org/ffmpeg.html)
#%% Test on one video

category = 'burying'
file_name = 'burying_1.gif'
path_2_file = path_input / category / file_name
out_file = path_output / (file_name[:-3] + 'mp4')

# ffprobe gathers information from multimedia streams and prints it in human-
# and machine-readable fashion.
output = str(subprocess.check_output(
  ['ffprobe', '-v', 'quiet', '-print_format', 'default', '-show_format', '-show_streams', str(path_2_file)]
  , stderr=subprocess.STDOUT)).split('\\n')
# -v quiet : set loglevel/verbose to silent
# -print_format default : set output printing format to default
#   For more details see: https://ffmpeg.org/ffprobe.html#toc-default
# -show_format : Show info about the container format of the input multimedia stream.
# -show_streams : Show info about each media stream contained in the input multimedia stream.

# Print height and width  of the input file
print(output[9], output[10])

# Execute conversion and see the output (if "0", then succesfull)
print(subprocess.call(
  ['ffmpeg', '-i', path_2_file,  '-movflags', 'faststart', '-pix_fmt', 'yuv420p', '-y', out_file]
  ))
# -i path_2_file : set input file from specific path
# -movflags faststart : Run a second pass moving the index (moov atom) to the beginning of the file.  (no idea what it is doing :-D)
# -pix_fmt yuv420p : set pixel format to “YUV” colour space with 4:2:0 chroma subsampling and planar colour alignment
# Source: https://ffmpeg.org/ffmpeg.html

# %% Run conversion on the acquired list (l_videos)
# Sweep through videos in list "l_videos"
start = time.time()

for i in range(len(l_videos)):
  # Verbose
  if i%50 == 0:
    print(f'{i}/{len(l_videos)}')

  category, file_name = l_videos[i]
  
  # Define input- & output-paths
  path_input_file = str(path_input / category/ file_name)
  path_output_file = str(path_output / category / (file_name[:-3] + 'mp4'))
  
  # Create output category directory if not present
  if not os.path.exists(path_output / category):
    os.mkdir(path_output / category)
    
  # Define shell command
  cmd = ['ffmpeg', '-i', path_input_file,  '-movflags', 'faststart',
         '-pix_fmt', 'yuv420p', '-y', path_output_file]
  
  # Run command and check if any error is encountered during conversion
  if subprocess.call(cmd) != 0:
    raise Exception(f'Error while converting {category}/{file_name}!')
    
stop = time.time()
duration = stop-start
print(f'\nTime elapsed: {duration:.2f}s (~ {duration/i:.3f}s per file)')
# %%
