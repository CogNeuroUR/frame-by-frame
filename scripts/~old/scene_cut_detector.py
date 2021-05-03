# [07.01.2020] OV
# Script writting GIFs 

################################################################################
# Imports
################################################################################
#%% Imports
from __future__ import print_function
# Path and sys related
import os
import pickle
from pathlib import Path
# Data types & numerics
import numpy as np
import pandas as pd
# Plots
import seaborn as sns
import matplotlib
## Change matplotlib backend to Qt
%matplotlib qt
# "%" specifies magic commands in ipython
import matplotlib.pyplot as plt
from PIL import Image

#%% Import custom utils
import sys, importlib
sys.path.insert(0, '..')
import utils
importlib.reload(utils) # Reload If modified during runtime

################################################################################
# Load full accuracy dictionary extracted w/ ResNet50-MiTv1
################################################################################
# %% Define paths
path_prefix = Path().parent.absolute()
dict_path = path_prefix / 'temp/full/accuracies_per_category_full_mitv1.pkl'
# Load from file
f = open(dict_path, 'rb')
accuracies_per_category = pickle.load(f)
# Print the categories
#print(len(list(accuracies_per_category.keys())))

#%% ############################################################################
# Extract best frame index from all available videos
################################################################################
# Load categories from txt file
import time

l_categories = utils.load_categories()
l_best = []

N = len(l_categories)
i = 0
for category in l_categories[:N]:
    print(i, category)
    i+=1
    # Get index of the category from l_categories (from 0 to 338)
    c_idx = [i for i in range(len(l_categories))
                if l_categories[i] == category][0]
    
    # draw random file from category
    l_files = list(accuracies_per_category[category].keys())
    
    if not l_files:
        continue
    else:
        for file_name in l_files:
            best, worst= utils.best_worst(accuracies_dict=accuracies_per_category,
                                           category_name=category,
                                           video_fname=file_name)
            l_best.append([category, file_name, best[0]])

############################## SCENE CUT DETECTION ############################
#%% Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.stats_manager import StatsManager
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

import cv2

def find_scenes(video_path, threshold=30.0, min_scene_len=5, verbose=False):
  # Create our video & scene managers, then add the detector.
  video_manager = VideoManager([video_path])
  # Get the number of frame
  n_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
  
  # Statistics
  stats_manager = StatsManager()
  #scene_manager = SceneManager()
  
  # Base timestamp at frame 0 (required to obtain the scene list).
  base_timecode = video_manager.get_base_timecode()

  # Improve processing speed by downscaling before processing.
  video_manager.set_downscale_factor()
  
  # Start the video manager and perform the scene detection.
  video_manager.start()
  scene_manager = SceneManager(stats_manager)
  # Add detector to scene manager
  scene_manager.add_detector(ContentDetector(threshold=threshold,
                                             min_scene_len=min_scene_len))
  # It detects FAST cuts using changes in colour and intensity between frames.
  
  # Detect scene changes with the chosen threshold
  scene_manager.detect_scenes(frame_source=video_manager, show_progress=False)
  
    # Sweep through frames and collect 'content_val' metric:
  l_content_vals = []
  for i in range(n_frames):
    l_content_vals.append(stats_manager.get_metrics(frame_number=i,
                                                    metric_keys=['content_val']))
  
  #print(l_content_vals)

  
  # Each returned scene is a tuple of the (start, end) timecode.
  #scene_list = scene_manager.get_scene_list(base_timecode)
  
  #<
  #video_manager.release()
  #>
  
  #return scene_list
  return scene_manager.get_scene_list(base_timecode), l_content_vals

#%%
# Detected for for threshold == 4
#category = 'praying'
#file_name = 'yt-YyFM-pWdqrY_311.mp4'

category = 'adult+female+speaking'
file_name = 'yt-aFk9L3nhS0Q_98.mp4'
path_to_csv = Path(f'outputs/{category}_scene_stats.csv')

path_2_file = Path(f'input_data/MIT_sampleVideos_RAW/{category}/{file_name}')
scenes, l_content_vals = find_scenes(str(path_2_file), 30, verbose=False)
print('\n')
print(scenes)

###############################################################################
#%% HERE LEFT 07.01
# Change matplotlib backend to Qt
%matplotlib qt
# "%" specifies magic commands in ipython
from scipy.stats import median_absolute_deviation
true_vals = np.array(l_content_vals[1:])
mad = median_absolute_deviation(true_vals)

estimate = []
errors = []

estimate.insert(0, true_vals[0][0])
errors.insert(0, 0)
for i in range(len(true_vals)):
  if i > 0 and i < len(true_vals) - 1:
    #estimate.append(np.median(true_vals[i-1:i+1]))
    estimate.append(np.mean(true_vals[i-1:i+1]))
    errors.append(np.array(estimate[i-1:i+1]).std())
    #estimate.append(median_absolute_deviation(true_vals[i-1:i+1]))
estimate.append(true_vals[-1][0])
errors.append(0)

estimate = np.array(estimate)
errors = np.array(errors)

x = np.linspace(0, len(true_vals), len(true_vals))
plt.plot(x, true_vals, color='black')
plt.plot(x, np.array(estimate), color='gray')
plt.fill_between(x, estimate-errors, estimate+errors, alpha=0.2)

plt.show() 

#%% Rolling mean and std
true_vals = np.array(l_content_vals[1:]).reshape((np.array(l_content_vals[1:]).shape[0],))
true_vals = pd.Series(true_vals)
#true_vals.plot(style='r')

#mean = true_vals.rolling(5, center=True, closed='both').mean() #.plot(style='k')
#mean = true_vals.rolling(5, center=True, closed='both').median() #.plot(style='k')
mean = true_vals.resample(rule='D').mean()

#std = mean.rolling(5, center=True, closed='both').std() #.plot(style='b')
std = true_vals.resample(rule='D').std()

plt.figure()
plt.plot(true_vals.index, true_vals)
plt.plot(mean.index, mean, c='gray')
plt.fill_between(std.index, mean-3*std, mean+3*std, color='g', alpha=0.2)
plt.show()

#%%

def main():
    #for num in [10, 50, 100, 1000]:
        # Generate some data
    x = np.array(l_content_vals[1:]).reshape((np.array(l_content_vals[1:]).shape[0],))

    plot(x)

    plt.show()

def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
  
def plot(x):
    fig, axes = plt.subplots(nrows=1)
    #for ax, func in zip(axes, [mad_based_outlier]):
    sns.distplot(x, ax=axes, rug=True, hist=False)
    outliers = x[mad_based_outlier(x)]
    axes.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    kwargs = dict(y=0.95, x=0.05, ha='left', va='top')
    axes.set_title('MAD-based Outliers', **kwargs)
    fig.suptitle(f'Outlier Tests ({category} : {file_name[:10]})', size=14)

main()

#%%
# From https://stackoverflow.com/a/22357811
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

#%%
true_vals = np.array(l_content_vals[1:]).reshape((np.array(l_content_vals[1:]).shape[0],))

bol = mad_based_outlier(true_vals)

print(np.where(bol == True))

#%%
cont_vals = np.array(l_content_vals, dtype=float)[1:]
print(cont_vals.shape)

cont_vals = cont_vals.reshape((cont_vals.shape[0],))
print(cont_vals.shape)

print(is_outlier(cont_vals))

#%%
fig, ax = plt.subplots()

outliers = cont_vals[is_outlier(cont_vals)]
idx_outliers = np.where(is_outlier(cont_vals) == True)[0]

plt.plot(cont_vals)
ax.plot(idx_outliers, outliers, 'ro', clip_on=False)
plt.show()

#%%
bol = np.where(is_outlier(cont_vals) == True)[0]
print(len(bol))
################################################################################
#%% TEST w/ is_outlier() on the subset
################################################################################
#%%
def extract_content(video_path, threshold=30.0, min_scene_len=5, verbose=False):
  # Create our video & scene managers, then add the detector.
  video_manager = VideoManager([video_path])
  # Get the number of frame
  n_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
  
  # Statistics
  stats_manager = StatsManager()
  #scene_manager = SceneManager()
  
  # Base timestamp at frame 0 (required to obtain the scene list).
  base_timecode = video_manager.get_base_timecode()

  # Improve processing speed by downscaling before processing.
  video_manager.set_downscale_factor()
  
  # Start the video manager and perform the scene detection.
  video_manager.start()
  scene_manager = SceneManager(stats_manager)
  # Add detector to scene manager
  scene_manager.add_detector(ContentDetector(threshold=threshold,
                                             min_scene_len=min_scene_len))
  # It detects FAST cuts using changes in colour and intensity between frames.
  
  # Detect scene changes with the chosen threshold
  scene_manager.detect_scenes(frame_source=video_manager, show_progress=False)
  
  # Sweep through frames and collect 'content_val' metric:
  l_content_vals = []
  for i in range(n_frames):
    l_content_vals.append(stats_manager.get_metrics(frame_number=i,
                                                    metric_keys=['content_val']))
  
  #print(l_content_vals)

  
  # Each returned scene is a tuple of the (start, end) timecode.
  #scene_list = scene_manager.get_scene_list(base_timecode)
  
  #<
  video_manager.release()
  #>
  
  #return scene_list
  return l_content_vals

#%%
############################## HERE LEFT 08.01.21 ###########################
#%% Tweaking ofthe threshold
start = time.time()
path_to_subset = Path('input_data/scene_cuts_subset/')
threshold = 6
l_outlaw = []

i = 0
for category, file_name, best in subset:
  #if category == 'adult+male+singing' and file_name == 'yt-5tdzrDI2ChE_300.mp4':
  if i % 20 == 0:
    print(f'{i}/{len(subset)}')
  i+=1
  # Load video file using decord
  path_out = path_to_subset / str(category + '__' + file_name)
  # Extract content_val metrics
  content_vals = np.array(extract_content(str(path_out)), dtype=float)[1:]
  # Reshape array
  content_vals = content_vals.reshape((content_vals.shape[0],))
  # Check for outliers
  if len(np.where(is_outlier(content_vals, threshold) == True)[0]) > 1:
    l_outlaw.append([category, file_name])

stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.3f}s per file)')
print(f'{len(l_outlaw)} were found w/ threshold={threshold}.')

for outlaw in l_outlaw:
  print(outlaw)

###############################################################################

#%% Test on a file
category = 'stitching'
file_name = 'yt-_1Y2p4bSwyA_144.mp4'

path_2_file = Path(f'input_data/MIT_sampleVideos_RAW/{category}/{file_name}')

if len(find_scenes(str(path_2_file), 40)) > 1:
    print('\n\nVideo DOES contain scene cuts!')
else:
    print('\n\nVideo does NOT contain scene cuts.')

#%% Test on multiple files
import cv2

pairs = []

start = time.time()

l_outlaw = []

i = 0
for category, file_name, best in l_best[:10]:
    if i % 20 == 0:
        print(f'\n{i}/{len(l_best)}')
    i+=1
    # Load video file using decord
    path_2_file = Path(f'input_data/MIT_sampleVideos_RAW/{category}/{file_name}')
    # Check
    if len(find_scenes(str(path_2_file), 40)) > 1:
        l_outlaw.append([category, file_name])
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.3f}s per file)')

################################################################################
#%% Tweak the threshold
################################################################################
#%% Collect a random subset of the video dataset
N = 100
temp = np.arange(len(l_best))
np.random.shuffle(temp)
subset_idxs = temp[:N]
subset = np.array(l_best)[subset_idxs]

# To be checked manually
for category, file_name, best in subset:
  print(category, file_name)

#%%
import pandas as pd
subset = pd.read_csv('outputs/video_subset_1.csv').to_numpy()
print(subset)

#%%
np.savetxt('outputs/video_subset.csv',subset,
           delimiter=',', fmt="%s",
           header=f'[07.01.21, OV] Subset')

#%% Copy the subset to a new directory
import shutil

path_to_subset = Path('input_data/scene_cuts_subset/')

for category, file_name, best in subset:
  path_to_input = Path(f'input_data/MIT_sampleVideos_RAW/{category}/{file_name}')
  path_out = path_to_subset / str(category + '__' + file_name)
  
  shutil.copyfile(src = path_to_input,
                  dst=path_out)
  

#%% Tweaking ofthe threshold
start = time.time()
path_to_subset = Path('input_data/scene_cuts_subset/')

threshold = 34
min_scene_len = 10
print('Threshold: ', threshold)
print('Min_scene_len: ', min_scene_len)
l_outlaw = []

i = 0
for category, file_name, best in subset:
  #if category == 'adult+male+singing' and file_name == 'yt-5tdzrDI2ChE_300.mp4':
  if i % 20 == 0:
    print(f'\n{i}/{len(l_best)}')
    #print(category, file_name)
  i+=1
  # Load video file using decord
  path_out = path_to_subset / str(category + '__' + file_name)
  # Check
  #print(find_scenes(str(path_2_file), threshold=threshold))
  if len(find_scenes(str(path_out),
                     threshold=threshold,
                     min_scene_len=min_scene_len)) > 1:
    l_outlaw.append([category, file_name])

stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.3f}s per file)')
print(f'{len(l_outlaw)} were found.')

for outlaw in l_outlaw:
  print(outlaw)

#%%


#%%
#%%
np.savetxt('outputs/outlaws_subset.csv',l_outlaw,
           delimiter=',', fmt="%s",
           header=f'[07.01.21, OV] PySceneDetect\nthreshold : {threshold}')

#%%

######################## SWEEEP through videos w/ scene cutter #################
#%%
import cv2

start = time.time()

l_fps = []
T = 0.5 # time duration of the wanted segments (in seconds) 

l_outlaw = []

i = 0
for category, file_name, best in l_best[:30]:
    if i % 20 == 0:
        print(f'\n{i}/{len(l_best)}')
    i+=1
    # Load video file using decord
    path_2_file = Path(f'input_data/MIT_sampleVideos_RAW/{category}/{file_name}')
    # Check
    if len(find_scenes(str(path_2_file), 40)) > 1:
        l_outlaw.append([category, file_name])

stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.3f}s per file)')

#%%
[print(outlaw) for outlaw in l_outlaw]

#%%
np.savetxt('outputs/outlaws_v2.csv',l_outlaw,
           delimiter=',', fmt="%s",
           header='[06.01.21, OV]\nList of videos that were detected to have \n \
                   scene cuts, using PySceneDetect.')

#%% Save frame differencing GIF
path_to_diff = Path('outputs/frame_differencing_drinking.gif')
utils.gif(path_to_diff, np.array(l_diffs), int(fps))
#%%
plt.imshow(l_diffs[6])
plt.savefig('outputs/2f_differencing_drinking_max.png')
plt.show()
# %%
plt.plot([i for i in range(len(l_diffs_sum))], np.gradient(l_diffs_sum))
plt.axhline(y=threshold, color='r', linestyle='-', label=f'Threshold ({threshold:.1E})')
plt.title('Gradients of the two-frame differences')
plt.legend()

#plt.savefig('outputs/2f_differencing_drinking_gradients.png')

plt.show()
#%% ############################################################################
# Interframe correlation for scenery change OR motion detection (TODO)
################################################################################
from scipy.signal import correlate
start = time.time()

l_fps = []
T = 0.5 # time duration of the wanted segments (in seconds) 
path_2_outputs = Path('outputs/GIFs_v1')

threshold = 25000000
l_outlaw = []

i = 0
for category, file_name, best in l_best[:500]:
    #if category == 'drinking' and file_name == 'yt-RS4tf6aXiQY_30.mp4':
    #if category == 'praying' and file_name == 'yt-YyFM-pWdqrY_311.mp4':
    #if category == 'burying' and file_name == 'yt-5sWtnzcJkww_219.mp4':
        # Verbose
        if i % 20 == 0:
            print(f'{i}/{len(l_best)}')
        i+=1
        # Load video file using decord
        path_2_file = Path(f'input_data/MIT_sampleVideos_RAW/{category}/{file_name}')
        vr = VideoReader(str(path_2_file))
        
        # Get FPS
        #l_fps.append(vr.get_avg_fps())
        #print(f'FPS: {vr.get_avg_fps()}')
        fps = vr.get_avg_fps()
        
        # Define the number of frames to be extracted
        N_frames = int(fps * T)
        
        # Define list with indices of frames ~ the position of the best frame
        #l_best_mid = [i for i in range(best - N_frames//2, best + N_frames//2)]
        l_best_mid = [i for i in range(best - N_frames, best)]
        
        # Empty arrays:
        frames_mid = np.array([])
        
        # Extract frames w/ decord
        if l_best_mid[-1] < vr.__len__():
            frames_mid = vr.get_batch(l_best_mid).asnumpy()
            
        # Compute three-frame differencing along the frames
        l_diffs_sum = []
        l_diffs = []
        for j in range(len(frames_mid)):
            if j > 0:
                frame1 = frames_mid[j-1]
                frame2 = frames_mid[j]
                diffs = cv2.absdiff(frame1, frame2)
                l_diffs_sum.append(np.sum(diffs))
                l_diffs.append(diffs)
        
        if np.absolute(np.gradient(l_diffs_sum)).max() > threshold:
            l_outlaw.append([category, file_name])
        
        """
        # Set every pixel that changed by 40 to 255, and all others to zero.
        threshold_value = 40
        set_to_value = 255
        result = cv2.threshold(diff, threshold_value, set_to_value, cv2.THRESH_BINARY)
        
        overlap = cv2.bitwise_and(difference_image_1, difference_image_2)
        """
 
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.3f}s per file)')