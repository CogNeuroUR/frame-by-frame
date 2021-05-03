# 16.04.21 OV
#
# Extraction of MIF indexes, based on RN50 softmax accuracy dictionary created
# using rn50_classification.ipynb

#%% Imports
# Path and sys related
import pickle
from pathlib import Path
# Data types & numerics
import pandas as pd
import numpy as np

#%% Import custom utils
import sys, importlib
sys.path.insert(0, '..')
import utils
# AB: What do they contain / what is needed here?
importlib.reload(utils) # Reload If modified during runtime

# %% Load full accuracy dictionary extracted w/ ResNet50-MiTv1
# AB: extracted at '01.mif_extraction_exploratory.ipynb' /  '01.mif_extraction_short.ipynb'
path_prefix = Path().parent.absolute()
dict_path = Path('../temp/accuracies_per_category_mitv1_fps-25.pkl')
# Load from file
f = open(dict_path, 'rb')
accuracies_per_category = pickle.load(f)

#%% Example of values
print(len(accuracies_per_category['singing']['yt-5xIQsJVNRz4_1227.mp4'][0]))

#%% Sweep through files in subfolders of path_input
import os
path_input = '../input_data/MIT_sampleVideos_RAW_final_25FPS'

l_videos = []
for path, subdirs, files in os.walk(path_input):
  for name in files:
    if name[-3:] == 'mp4': # AB: file =? .MP4 (ß)
      l_videos.append([path.split('/')[-1],   # category
                       name])                 # file name
    else:
      print('Ignored: ', name)

if l_videos:
  l_videos = sorted(l_videos)

# Number of MP4s found:
print('Total nr. of MP4s: ', len(l_videos))

#%% Load RN50 softmax dictionary
path_labels = '/models/labels/category_momentsv1.txt'

# load categories
categories = utils.load_categories(path_labels)

#%% Example 
category = 'aiming'
file_name = 'yt-0gwUV4Ze-Hs_390.mp4'
cat_idx = categories.index(category)
pred_accuracies = np.array(accuracies_per_category[category][file_name])[:, cat_idx]

#%% Extract MIF idx-s
l_mifs = []

l_notfound = []

for i in range(len(l_videos)):
  # Get category and file names
  category, file_name = l_videos[i]
  #print(category, file_name)
  # Get idx. of true category from original ordering of categories
  cat_idx = categories.index(category)

  # Extract softmax values for the true category, i.e. cat_idx
  # AB: even if it is not the top entry (or not even in the top 5)
  # Check if values exists:
  try:
    pred_accuracies = np.array(accuracies_per_category[category][file_name])[:, cat_idx]
    #break
    #print(pred_accuracies.shape)
    # Append to output list of mifs (:= most informative frames)
    l_mifs.append([category, file_name,
                  np.argmax(pred_accuracies),
                  pred_accuracies[np.argmax(pred_accuracies)]])
  except KeyError:
    l_notfound.append([category, file_name])
    print('OOOps, error at ', print(category, file_name))
  
print('Found: ', len(l_mifs))
print('Not found: ', len(l_notfound))

#%% Convert to pd.DataFrame
df = pd.DataFrame(l_mifs, columns=['category', 'fname', 'mif_idx', 'softmax[category]'])
print(df)

#%% Save to .csv
df.to_csv('../temp/mifs_MIT_sampleVideos_RAW_final_25FPS_from-dict.csv')
