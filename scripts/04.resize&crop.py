#%%#############################################################################
# OV 01.04.21
# Make sure to switch to stock (3.8.2) python which has access to homebrew
# installation of the ffmpeg -> lets us encode back into h264 (no visual loss)
################################################################################
#%% Imports 
from pathlib import Path
import time
import cv2
import subprocess
import os

#%% Sweep through videos
# AB: change paths if necessary (might be run locally / dependencies (!))
path_input = Path('../input_data/MIT_additionalVideos_25FPS/').absolute()
path_output = Path('../input_data/MIT_additionalVideos_25FPS_480x360p').absolute()

if not os.path.exists(path_output):
  os.makedirs(path_output)

#%% Sweep through files in subfolders of path_input
l_videos = []
for path, subdirs, files in os.walk(path_input):
  for name in files:
    # Check if files have .mp4 extension
    if name[-3:] == 'mp4':
      l_videos.append([path.split('/')[-1],   # category
                       name])                 # file name
    # Report non-mp4 files
    else:
      print('Ignored: ', name)

if l_videos:
  l_videos = sorted(l_videos)
# AB: Terminal Feedback to User
print('Total nr. of MP4s: ', len(l_videos))

#%% ############################################################################
# Resizing of the videos
################################################################################
#%% Test on one video
width = 480
#category = 'bicycling'
#file_name = 'yt-5r5WH6nBey8_235.mp4'
category = 'burying'
file_name = 'yt-_6awwB9VXzo_13.mp4'
path_2_file = str(Path(f'../input_data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}').absolute())
out_file = path_output / file_name

output = str(subprocess.check_output(
  ['ffprobe', '-v', 'quiet', '-print_format', 'default', '-show_format', '-show_streams', path_2_file]
  , stderr=subprocess.STDOUT)).split('\\n')
  # AB: Please add some info (or reference to man page) on parameters here
  # AB: This represents the core of what is done in the script, I guess
print(output[9][6:], output[10][7:])

print(subprocess.call(
  ['ffmpeg', '-i', path_2_file,  '-vcodec', 'libx264', '-vf', f'scale={width}:-2', '-y', out_file]
  ))

#%% Run a subset
start = time.time()

# Parameters: ==============================================
out_width = 480 #mean_width
# ==========================================================   

# Sweep through videos in list
for i in range(len(l_videos)):
  # Verbose
  if i%50 == 0:
    print(f'{i}/{len(l_videos)}')

  category, file_name = l_videos[i]
  
  # Define input- & output-paths
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
  
  # Define shell command
  cmd = ['ffmpeg', '-i', path_input_file, '-vcodec', 'libx264',
      '-filter:v', f"crop={shapes}, scale={scales}", '-y', path_output_file]
  
  # Run command
  if subprocess.call(cmd) != 0:
    raise Exception(f'Error while resizing {category}/{file_name}!')
    
stop = time.time()
duration = stop-start
print(f'\nTime elapsed: {duration:.2f}s (~ {duration/i:.3f}s per file)')

# %% Test each output video if has the correct size (w/ decord)
# AB: Why imports here and not exclusively above?
import cv2
import subprocess

for category, file_name in l_videos:
  # Define path
  path_input_file = str(path_output / category / file_name)
  path_output_file = str(path_output / category / 'temp.mp4')
  
  # Create output category directory if not present
  if not os.path.exists(path_input_file):
    print(category, file_name, ' does NOT exist!')
  else:
    vcap = cv2.VideoCapture(path_input_file)
    width  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Exact cropping command
    cmd = ['ffmpeg', '-i', path_input_file, '-vcodec', 'libx264',
           '-filter:v', 'crop=480:360', '-y', path_output_file]
    if width != 480:
      print(category, file_name, f' has wrong width ({width})!')
      
      print('\tCroping to exactly 480x360p...')
      if subprocess.call(cmd) != 0:
        raise Exception(f'Failed croping!')
      
      # Remove input and rename temp to input
      os.remove(path_input_file)
      os.rename(path_output_file, path_input_file)
    
    if height != 360:
      print(category, file_name, f' has wrong height ({height})!')
      
      print('\tCroping to exactly 480x360p...')
      if subprocess.call(cmd) != 0:
        raise Exception(f'Failed croping!')
      
      # Remove input and rename temp to input
      os.remove(path_input_file)
      os.rename(path_output_file, path_input_file)
    
print('Test finished!')

# %% For statistics
# AB: change path if necessary
path_output = Path('../input_data/MIT_additionalVideos_25FPS_480x360p').absolute()
l_processed = []
for path, subdirs, files in os.walk(path_output):
  for name in files:
    if name[-3:] == 'mp4':
      l_processed.append([path.split('/')[-1],   # category
                       name])                    # file name
    else:
      print('Ignored: ', name)

if l_processed:
  l_processed = sorted(l_processed)
print('Total nr. of MP4s: ', len(l_processed))

# AB: What is accomplished in this block?

# %%
