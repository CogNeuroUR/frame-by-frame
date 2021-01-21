# [16.12.20] OV
# Script to change framerate of videos

################################################################################
# Imports
################################################################################
#%%
import time
import ffmpeg
from pathlib import Path
import pickle
import os

#%% Import custom utils
import sys, importlib
sys.path.insert(0, '..')
import utils
importlib.reload(utils) # Reload If modified during runtime

################################################################################
#%% Sweep through videos
################################################################################
#%% Paths
path_test = Path('data/test')
dict_path = Path('saved/full/accuracies_per_category_full_mitv1.pkl')
# Load from file
f = open(dict_path, 'rb')
accuracies_per_category = pickle.load(f)
l_categories = utils.load_categories()


#%%
path_outputs = Path('data/MIT_sampleVideos_25FPS/')
if not os.path.exists(path_outputs):
  os.makedirs(path_outputs)

#%%
i_fps = 25
keep_audio = True

start = time.time()

def HasAudioStreams( file_path ):
    streams = ffmpeg.probe(file_path)["streams"]
    for stream in streams:
        if stream["codec_type"] == "audio":
            return True
    return False

j = 0
i = 0
for category in l_categories:
    # Verbose
    print(f'{j}/{len(l_categories)}'); j+=1
    # Load video file using decord
    l_files = list(accuracies_per_category[category].keys())
    
    # Check if category directory exists
    path_output_category = Path(f'data/MIT_sampleVideos_25FPS/{category}')
    if not os.path.exists(path_output_category):
      os.makedirs(path_output_category)
    
    # Sweep through file in category
    for file_name in l_files:
      # Define paths to input and output files
      path_input_file = str(Path(f'data/MIT_sampleVideos_RAW_test/{category}/{file_name}'))
      path_output_file = str(path_outputs / category / file_name)
      #print(path_output_file)
      
      if keep_audio == False: # Do not keep audio
        (
            ffmpeg
            .input(str(path_input_file))
            .filter('fps', fps=i_fps, round='near')
            .output(str(path_output_file))
            .overwrite_output()
            .run()
        )
      else: # Keep audio
        try:
          # Load file
          in_ = ffmpeg.input(path_input_file)
          # Extract video and audio streams separately
          a = in_.audio if HasAudioStreams(path_input_file) else None
          #print(a.)
          v = in_.video.filter('fps', fps=i_fps, round='near') # change FPS
          # Save to output file, keeping the audio stream unchanged
          if a:
            out = ffmpeg.output(v, a, path_output_file, acodec='copy')
          else:
            out = ffmpeg.output(v, path_output_file, acodec='copy')
          out.run(capture_stdout=True,
                  capture_stderr=True,
                  overwrite_output=True)
        except ffmpeg.Error as e:
          print('stdout:', e.stdout.decode('utf8'))
          print('stderr:', e.stderr.decode('utf8'))
          raise e
      # Increment
      i+=1
    
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/i:.2f}s per file)')

#%% Sweep
# %%
