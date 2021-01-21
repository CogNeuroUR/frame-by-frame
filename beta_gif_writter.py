# [07.12.2020] OV
# Script writting GIFs 

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
from sklearn.cluster import AgglomerativeClustering
# Plots
import seaborn as sns
import matplotlib
#%matplotlib qt
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
dict_path = path_prefix / 'saved/full/accuracies_per_category_full_mitv1.pkl'
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

#%% Collect FPS-es
import cv2
start = time.time()
# Parameters: ==============================================
l_fps = []
l_high_fps = []
# ==========================================================

i = 0
for category, file_name, best in l_best:
    # Verbose
    print(f'{i}/{len(l_best)}')
    i+=1
    # Load video file using decord
    path_2_file = Path(f'data/MIT_sampleVideos_RAW/{category}/{file_name}')
    vr = VideoReader(str(path_2_file))
    
    # Get FPS
    l_fps.append(vr.get_avg_fps())
    if vr.get_avg_fps() > 30:
        l_high_fps.append([category, file_name, vr.get_avg_fps()])
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.2f}s per file)')

#%%
np.savetxt('outputs/high_fps_videos.csv',l_high_fps,
           delimiter=',', fmt="%s",
           header='[08.12.20, OV]\nList of videos FPS>30')


#%% ############################################################################
# Collect frames by the extracted indexes
################################################################################
import cv2
start = time.time()
# Parameters: ==============================================
l_fps = []
T = 1.0 # time duration of the wanted segments (in seconds) 
# ==========================================================
if T == 0.5:
    path_2_outputs = Path('outputs/GIFs_v1')
elif T == 1.0:
    path_2_outputs = Path('outputs/GIFs_v1_1s')
else:
    raise Exception('Sequence length (T) different thant 0.5 or 1.0 seconds!')
save_all_three = True
# ==========================================================

i = 0
for category, file_name, best in l_best[150:180]:
    # Verbose
    print(f'{i}/{len(l_best)}')
    i+=1
    # Load video file using decord
    path_2_file = Path(f'data/MIT_sampleVideos_RAW/{category}/{file_name}')
    vr = VideoReader(str(path_2_file))
    
    # Get FPS
    #l_fps.append(vr.get_avg_fps())
    #print(f'FPS: {vr.get_avg_fps()}')
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
    start_path = Path(f'{path_2_outputs}/{category}/{file_name}_start.gif')
    mid_path = Path(f'{path_2_outputs}/{category}/{file_name}_mid.gif')
    end_path = Path(f'{path_2_outputs}/{category}/{file_name}_end.gif')
    
    # Check if category path exist, if not, make one
    os.makedirs(f'{path_2_outputs/category}', exist_ok=True)
    
    ###################################
    # The framerate changes the duration of the clip .....
    #fps = 10
    
    #########################################
    
    # Save arrays to gifs:
    # IF "best-in-the-middle" exists, save it, otherwise, go to "...-begin", ...
    if save_all_three == False: # Save only "mid" OR "begin" OR "end"
        if frames_mid.size != 0:
            utils.gif(mid_path, frames_mid, int(fps))
        elif frames_begin.size != 0:
            utils.gif(start_path, frames_begin, int(fps))
        else:
            utils.gif(end_path, frames_end, int(fps))
    else: # OR save all three (if not empty)
        if frames_begin.size != 0:
            utils.gif(start_path, frames_begin, int(fps))
        if frames_mid.size != 0:
            utils.gif(mid_path, frames_mid, int(fps))
        if frames_end.size != 0:
            utils.gif(end_path, frames_end, int(fps))

stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.4f}s per file)')

# %% Plot distribution of FPSs
%matplotlib qt
sns.displot(l_fps, bins=20)
plt.title('Distribution of Framerates over the videos')
plt.tight_layout()
plt.yscale('log')
plt.show()

#%% ############################################################################
# Two-frame differencing for motion detection
################################################################################
import cv2

start = time.time()

l_fps = []
T = 0.5 # time duration of the wanted segments (in seconds) 
path_2_outputs = Path('outputs/GIFs_v1')

threshold = 25000000
l_outlaw = []

i = 0
for category, file_name, best in l_best:
    #if category == 'drinking' and file_name == 'yt-RS4tf6aXiQY_30.mp4':
    if category == 'praying' and file_name == 'yt-YyFM-pWdqrY_311.mp4':
    #if category == 'burying' and file_name == 'yt-5sWtnzcJkww_219.mp4':
    #if category == 'slapping' and file_name == 'yt-_drWwTZgj6o_603.mp4':
        # Verbose
        if i % 20 == 0:
            print(f'{i}/{len(l_best)}')
        i+=1
        # Load video file using decord
        path_2_file = Path(f'data/MIT_sampleVideos_RAW/{category}/{file_name}')
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
            
        # Compute two-frame differencing along the frames
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

#%%
print(len(l_outlaw))
print(l_outlaw)

#%%
np.savetxt('outputs/outlaws_v1.csv',l_outlaw,
           delimiter=',', fmt="%s",
           header='[08.12.20, OV]\nList of videos that crossed the custom \n \
               threshold of 25e6, computed by this formula: abs(max(grad(frame-to-frame differences)))')

#%% Compute variance
mean_diff = np.array(l_diffs).mean(axis=0)
diffs = np.array(l_diffs)
var = []

for i in range(len(diffs)):
    if i > 1:
        #mean = np.array(l_diffs[i-1 : i]).mean(axis=0)
        #var.append(np.square(l_diffs[i-1] - mean)
        #           + np.square(l_diffs[i] - mean))
        var.append(np.var(diffs[i-2 : i]))
        
for i in range(len(var)):
    print(i, var[i])
# Compute gradients
grads = np.array(np.gradient(diffs, axis=0))

#%% Plot gradients
%matplotlib qt
import matplotlib.animation as animation
fig, ax = plt.subplots()
ims = []
for frame in grads:
    im = plt.imshow(frame, animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                repeat_delay=200)
plt.show()


############################## SCENE CUT DETECTION ############################
#%% Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

def find_scenes(video_path, threshold=30.0, min_scene_len=2, verbose=False):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    
    #<
    # Statistics
    stats_manager = StatsManager()
    #scene_manager = SceneManager()
    scene_manager = SceneManager(stats_manager)
    #>
    
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    # It detects FAST cuts using changes in colour and intensity between frames.
    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()
    
    #<
    # Verbose : print stats
    if verbose:
        print(scene_manager.stat)
    #>
    
    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list(base_timecode)

#%%
# Detected for for threshold == 4
#category = 'praying'
#file_name = 'yt-YyFM-pWdqrY_311.mp4'

category = 'slapping'
file_name = 'yt-_drWwTZgj6o_603.mp4'
path_2_file = Path(f'data/MIT_sampleVideos_RAW/{category}/{file_name}')
scenes = find_scenes(str(path_2_file), 40)
print('\n')
print(scenes)

#%% Test on a file
category = 'stitching'
file_name = 'yt-_1Y2p4bSwyA_144.mp4'

path_2_file = Path(f'data/MIT_sampleVideos_RAW/{category}/{file_name}')

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
for category, file_name, best in l_best:
    if i % 20 == 0:
        print(f'\n{i}/{len(l_best)}')
    i+=1
    # Load video file using decord
    path_2_file = Path(f'data/MIT_sampleVideos_RAW/{category}/{file_name}')
    # Check
    if len(find_scenes(str(path_2_file), 40)) > 1:
        l_outlaw.append([category, file_name])
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.3f}s per file)')

######################## SWEEEP through videos w/ scene cutter #################
#%%
import cv2

start = time.time()

l_fps = []
T = 0.5 # time duration of the wanted segments (in seconds) 

l_outlaw = []

i = 0
for category, file_name, best in l_best:
    if i % 20 == 0:
        print(f'\n{i}/{len(l_best)}')
    i+=1
    # Load video file using decord
    path_2_file = Path(f'data/MIT_sampleVideos_RAW/{category}/{file_name}')
    # Check
    if len(find_scenes(str(path_2_file), 40)) > 1:
        l_outlaw.append([category, file_name])

stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.3f}s per file)')

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
        path_2_file = Path(f'data/MIT_sampleVideos_RAW/{category}/{file_name}')
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