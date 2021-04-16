# 14.04.21 OV
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
importlib.reload(utils) # Reload If modified during runtime

# %% Load full accuracy dictionary extracted w/ ResNet50-MiTv1
path_prefix = Path().parent.absolute()
dict_path = Path('../saved/accuracies_per_category_mitv1_fps-25.pkl')
# Load from file
f = open(dict_path, 'rb')
accuracies_per_category = pickle.load(f)

#%%
print(len(accuracies_per_category['adult+female+singing']['yt-5xIQsJVNRz4_1227.mp4'][0]))

#%%

#%% Sweep through files in subfolders of path_input
import os
path_input = '../data/MIT_sampleVideos_RAW_final'

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

#%% Load RN50 softmax dictionary
path_labels = '../labels/category_momentsv1.txt'

# load categories
categories = utils.load_categories(path_labels)

#%%
cat_idx = categories.index('adult+female+singing')
pred_accuracies = np.array(accuracies_per_category['adult+female+singing']['yt-0rvJOREmAv4_60.mp4'])[:, cat_idx]

#%% Extract MIF idx-s
l_mifs = []

l_notfound = []

for i in range(len(l_videos[:10])):
  # Get category and file names
  category, file_name = l_videos[i]
  print(category, file_name)
  # Get idx. of true category from original ordering of categories
  cat_idx = categories.index(category)

  # Extract softmax values for the true category, i.e. cat_idx
  # Check if values exists:
  try:
    pred_accuracies = np.array(accuracies_per_category[category][file_name])[:, cat_idx]
    #break
    print(pred_accuracies.shape)
  except KeyError:
    print('OOOps')
  
  """
  if file_name in accuracies_per_category.values():
    
    # Append to output list
    l_mifs.append([category, file_name,
                  np.argmax(pred_accuracies),
                  pred_accuracies[np.argmax(pred_accuracies)]])
  else:
    #print('Values not found at ', category, file_name)
    l_notfound.append([category, file_name])
  """
  
print('Found: ', len(l_mifs))
print('Not found: ', len(l_notfound))

#%% Save to .csv
df = pd.DataFrame(l_mifs, columns=['category', 'fname', 'mif_idx', 'softmax[category]'])
print(df)
df.to_csv(path_prefix / 'saved/mifs_from-dict.csv')