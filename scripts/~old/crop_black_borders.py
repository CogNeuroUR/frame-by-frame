# [18.01.21] OV
################################################################################
# Imports
################################################################################
#%% Imports
from __future__ import print_function
# Image/video manipulation
import decord
decord.bridge.set_bridge('native')
from decord import VideoReader
from decord import cpu
from PIL import Image, ImageChops
import cv2
import ffmpeg
# Path and sys related
import os
import sys
import pickle
import time
import subprocess
from pathlib import Path
import re
# Data types & numerics
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
# Plots
import seaborn as sns
import matplotlib
#%matplotlib qt
import matplotlib.pyplot as plt

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

#%% Order alphabetically the categories
l_sorted_best = sorted(l_best)
print(l_sorted_best[:10])

#%% ############################################################################
# Cropping black borders using ffmpeg
################################################################################
#%% Test file
#category = 'assembling'
#fname = 'yt-vJbPD1EPAgQ_1677.mp4'
category = 'attacking'
fname = 'meta-476930_24.mp4'
in_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{fname}'))
out_file = str(Path(f'data/cropping_tests/{category}_{fname}'))

print(subprocess.call(['ffmpeg', '-i', in_file, '-vf', 'cropdetect=24:16:0', out_file]))
#%%
#crop_dimensions = subprocess.Popen(['ffmpeg -i ' + in_file + ' -vf cropdetect -f null - 2>&1 | awk \'/crop/ { print $NF }\' | tail -1'], shell=True, stdout=subprocess.PIPE).stdout.read().strip()
md = ['ffmpeg', '-i', str(in_file), '-t', '1', '-vf', 'cropdetect',
       '-f', 'null', '-', '2>&1', '|', 'awk', '\'/crop/ { print $NF }\'', '|', 'tail', '-1']
print(cmd)

print(subprocess.call(cmd))
c
"""
cmd = [
    'ffmpeg', 
    '-y',
    '-i', in_file,
    '-vf', crop_dimensions,
    out_file
]
print(cmd)
subprocess.Popen(cmd, stdout = subprocess.PIPE, bufsize=10**8)
"""

#%%
def detectCropFile(input_fname):#, output_fname):
    fpath = str(input_fname)
    print("File to detect crop: %s " % fpath)
    p = subprocess.Popen(["ffmpeg", "-i", fpath, "-vf", "cropdetect=24:16:0", "-vframes", "500", "-f", "rawvideo", "-y", "/dev/null"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    infos = p.stderr.read()
    print(infos)
    allCrops = re.findall(CROP_DETECT_LINE + ".*", infos)
    print(allCrops) 
    mostCommonCrop = Counter(allCrops).most_common(1)
    print("most common crop: %s" % mostCommonCrop)
    print(mostCommonCrop[0][0])
    """
    global crop
    crop = mostCommonCrop[0][0]
    video_rename()
    """

def detectCrop(input_fname):#, output_fname):
    fpath = str(input_fname)
    print("File to detect crop: %s " % fpath)
    result = subprocess.check_output(["ffmpeg", "-i", fpath, "-vf", "cropdetect=24:16:0", "-vframes", "500", "-f", "rawvideo", "-y", "/dev/null"])
    # this regex matches the string crop followed by one or more non-whitespace characters 
    match = re.search("crop\S+", result) 
    crop_result = None
    if match is not None:
     crop_result = match.group()
    """
    p = subprocess.Popen(["ffmpeg", "-i", fpath, "-vf", "cropdetect=24:16:0", "-vframes", "500", "-f", "rawvideo", "-y", "/dev/null"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    infos = p.stderr.read()
    print(infos)
    allCrops = re.findall(CROP_DETECT_LINE + ".*", infos)
    print(allCrops) 
    mostCommonCrop = Counter(allCrops).most_common(1)
    print("most common crop: %s" % mostCommonCrop)
    print(mostCommonCrop[0][0])
    """
    """
    global crop
    crop = mostCommonCrop[0][0]
    video_rename()
    """

category = 'assembling'
fname = 'yt-vJbPD1EPAgQ_1677.mp4'
in_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{fname}').absolute())


detectCropFile(in_file)

#%% ############################################################################
# Search for (black) border containing videos
################################################################################
#%% Function definition
def has_border(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    #diff = ImageChops.add(diff, diff)      
    bbox = diff.getbbox()
    return bbox != (0,0,im.size[0],im.size[1]), bbox
  
def return_border(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    #diff = ImageChops.add(diff, diff)      
    bbox = diff.getbbox()
    return bbox
  
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def trim_cv(filename):
  im = cv2.imread(filename)
  h,w,d = im.shape
  #left limit
  for i in range(w):
      if np.sum(im[:,i,:]) > 0:
          break
  #right limit
  for j in xrange(w-1,0,-1):
      if np.sum(im[:,j,:]) > 0:
          break

  cropped = im[:,i:j+1,:].copy()

#%% Test on a file (frame by frame)
#category = 'assembling'
#fname = 'yt-vJbPD1EPAgQ_1677.mp4'
category = 'attacking'
fname = 'meta-476930_24.mp4'
in_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{fname}'))
out_file = str(Path(f'data/cropping_tests/{category}_{fname}'))

# Load video
cap = cv2.VideoCapture(str(in_file))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#codec = cv2.VideoWriter_fourcc(*'MP4V')
codec = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(out_file, codec, fps, (width,  height))

if os.path.exists(in_file):
  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print('Nr. of frames: ', n_frames)
  first = True
  for i in range(n_frames):
    _, frame = cap.read()
    img_frame = Image.fromarray(frame)
    #plt.imshow(frame)
    #plt.show()
    
    if has_border(img_frame):
      print(i, True)
      frame = trim(img_frame)
      
      out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
      if first:
        #img_frame.show()
        #trimmed.show()
        first = False
      #l_borders.append([category, file_name]) 
      
  # Release everything if job is finished
  cap.release()
  out.release()
else:
  print('Not found!') 

#%% Test on a file
#category = 'assembling'
#fname = 'yt-vJbPD1EPAgQ_1677.mp4'
category = 'attacking'
fname = 'meta-476930_24.mp4'
in_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{fname}'))
out_file = str(Path(f'data/cropping_tests/{category}_{fname}'))

# Load video
cap = cv2.VideoCapture(str(in_file))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#codec = cv2.VideoWriter_fourcc(*'MP4V')
codec = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(out_file, codec, fps, (width,  height))

min_rectangle = [width, height]

if os.path.exists(in_file):
  n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print('Nr. of frames: ', n_frames)
  first = True
  for i in range(n_frames):
    _, frame = cap.read()
    img_frame = Image.fromarray(frame)
    #plt.imshow(frame)
    #plt.show()
    
    truth, bbox = has_border(img_frame)
    if min_rectangle[0] > bbox[2]:
      min_rectangle[0] = bbox[2]
    elif min_rectangle[1] > bbox[3]:
      min_rectangle[1] = bbox[3]
    if truth:
      print(i, min_rectangle)
      
  # Release everything if job is finished
  cap.release()
  out.release()
else:
  print('Not found!') 

#%%
#%% Sweep through videos

start = time.time()
# Parameters: ==============================================
l_borders = []
# ==========================================================

def has_border(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    #diff = ImageChops.add(diff, diff, 2.0, -100)
    diff = ImageChops.add(diff, diff)      
    bbox = diff.getbbox()
    return bbox != (0,0,im.size[0],im.size[1])

i = 0
l_cats = [] 
for category, file_name, best in l_sorted_best:
    # Verbose 
    print(f'{i}/{len(l_sorted_best)}')
  
    
    # Load video file using decord
    path_2_file = Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}')
    
    # Check if file exists:
    if os.path.exists(path_2_file):
      #vr = VideoReader(str(path_2_file))
      cap = cv2.VideoCapture(str(path_2_file))
      
      # Get sizes
      #frame = vr.get_batch([0]).asnumpy()[0]
      _, frame = cap.read()
      """
      #img = cv2.imread('sofwin.png')
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

      contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      cnt = contours[0]
      x,y,w,h = cv2.boundingRect(cnt)
      """
      #print(f'{category}/{file_name}')
      #print(frame.shape)
      #print(has_border(Image.fromarray(frame)))
      
      """
      plt.imshow(frame)
      plt.show()
      """
      if has_border(Image.fromarray(frame)):
        l_borders.append([category, file_name])
          
    
      i+=1
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.2f}s per file)')

# %%
print(l_borders)

# %%
bordered_df = pd.DataFrame(l_borders, columns=['Categories', 'File names'])
print(bordered_df)
#%%
bordered_df.to_csv('outputs/RAW_DOWNSIZING_IN_PROGRESS_bordered_videos.csv')

################################################################################
# %% Square cropping
################################################################################
# Based on https://video.stackexchange.com/a/4571

width = 899
height = 899
category = 'bicycling'
file_name = 'yt-5r5WH6nBey8_235.mp4'
path_2_file = str(Path(f'data/resizing_tests/width_480/w480_{category}_{file_name}').absolute())
out_file = str(Path(f'data/resizing_tests/square_cropped/sq_cr_w480_{category}_{file_name}').absolute())

if os.path.exists(path_2_file):
  subprocess.call(
    #['ffmpeg', '-i', path_2_file,  '-filter:v', f'crop={width}:{height}', '-c:a', 'copy', out_file])
    ['ffmpeg', '-i', path_2_file,  '-vf', 'crop=in_h:in_h', '-c:a', 'copy', out_file])
  print('file %s saved' % out_file)
else:
  print('No file found!')

################################################################################
#%% Test on a subset
N = 20
temp = np.arange(len(l_best))
np.random.shuffle(temp)
subset_idxs = temp[:N]
subset = np.array(l_best)[subset_idxs]

for category, file_name, best in subset:
  print(category, file_name)

#%% Start square cropping  
start = time.time()
# Parameters: ==============================================
out_folder = Path(f'data/cropping_tests/comparison/').absolute()
# ==========================================================

i = 0
l_cats = [] 
for category, file_name, best in subset:
  # Verbose 
  print(f'{i}/{len(subset)}')

  path_2_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}').absolute())
  output_file = str(out_folder / f'sqcr_{category}_{file_name}')
  
  if os.path.exists(path_2_file):
    subprocess.call(
      ['ffmpeg', '-i', path_2_file,  '-vf', 'crop=in_h:in_h', '-c:a', 'copy', output_file])
    print('file %s saved' % output_file)
  else:
    print('No file found!')

  i += 1
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/N:.3f}s per file)')

#%% Copy the subset for comparison
import shutil
i=0
for category, file_name, best in subset:
    # Verbose 
    print(f'{i}/{len(subset)}')
  
    # Load video file using decord
    path_2_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}').absolute())    
    # Get width
    if os.path.exists(path_2_file):
        vr = VideoReader(str(path_2_file))
        original_width = vr.get_batch([0]).shape[1]
        
        output_file = str(out_folder / f'ls_{original_width}_{category}_{file_name}')
        
        shutil.copyfile(src=path_2_file,
                        dst=output_file)
    i += 1

#%% Center-cropping function
def crop_center(img,cropx,cropy):
    y, x, _ = img.shape
    if y < x: # landscape
      startx = x//2-(cropx//2)
      starty = y//2-(cropy//2)
    else:
      y, x = x, y
      startx = x//2-(cropx//2)
      starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx, :]

# %% Save frames

# Parameters: ==============================================
out_folder = Path(f'data/cropping_tests/comparison/frames/').absolute()
arr_best = np.array(l_best)
crop = False#True
# ==========================================================

i = 0
for category, file_name, best in subset:
  # Verbose 
  print(f'{i}/{len(subset)}')

  # Load video file using decord
  path_2_file = str(Path(f'data/MIT_sampleVideos_RAW_DOWNSIZING_IN_PROGRESS/{category}/{file_name}').absolute())    
  # Check if file exists:
  if os.path.exists(path_2_file):
    #vr = VideoReader(str(path_2_file))
    cap = cv2.VideoCapture(str(path_2_file))
    
    # Find best frame idx
    src_result = np.where(arr_best[:, 0:2] == [category, file_name])
    index = src_result[0][np.where(src_result[1] == 1)]
    best_frame_idx = arr_best[index][0][2]
    
    # cap.set(1, wanted_frame_idx) where "1" stands for CV_CAP_PROP_POS_FRAMES 
    cap.set(1, int(best_frame_idx))
    
    _, frame = cap.read()
    file_name = file_name[:-3] + 'jpg'
    print('Before: ', frame.shape)
    if crop:
      height = min(frame.shape[:2])
      print('height=', height)
      frame = crop_center(frame, height, height)
      output_file = str(out_folder / f'cropped_{category}_{file_name}')
    else:
      output_file = str(out_folder / f'original_{category}_{file_name}')
    print('After: ', img_frame.size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_frame = Image.fromarray(frame, mode='RGB')
    
    #img_frame.show()
    img_frame.save(output_file, mode='RGB')
    i += 1
#%%
l_best.index(['bicycling', 'yt-5r5WH6nBey8_235.mp4', :])

# %% Search for best frame
arr_best = np.array(l_best)
src_result = np.where(arr_best[:, 0:2] == ['bicycling', 'yt-5r5WH6nBey8_235.mp4'])
index = src_result[0][np.where(src_result[1] == 1)]
best_frame_idx = arr_best[index][0][2]
print(best_frame_idx)
# %%
