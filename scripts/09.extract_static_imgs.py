# [14.04.21] OV
# Script extracting static images, i.e MIFs

################################################################################
# Imports
################################################################################
#%% Imports
from __future__ import print_function
# Path and sys related
import os
from pathlib import Path
# Data types & numerics
import pandas as pd
import matplotlib.pyplot as plt
import PIL # AB: causes errors at mi Win: 'Unable to import 'PIL'pylint(import-error)'
# AB: What is PIL used for (would help understanding)

#%% Load MIF csv
# AB: a bit more info abt what is stored in the csv (1 line)
# change path if necessary
path_mifs = Path('../saved/mifs.csv')
mifs = pd.read_csv(path_mifs, usecols=['category', 'fname', 'mif_idx'])
print(mifs)
  
#%% ############################################################################
# Test on a single video
################################################################################
category = 'arresting'
file_name = 'yt-aAVfUYxx12g_18.mp4'
path_2_file = Path(f'..data/single_video_25FPS_480x360p/{category}/{file_name}')

best = mifs.loc[(mifs['category'] == category) & (mifs['fname'] == file_name)]['mif_idx'].values[0]
print('Best frame idx: ', best)
vr = VideoReader(str(path_2_file)) # AB: is VideoReader loaded correctly; displays for me as not defined
mif = vr[best].asnumpy()
print(mif.shape)

plt.imshow(mif)
plt.show()

im = PIL.Image.fromarray(mif)
im.save(Path(f'data/single_video_25FPS_480x360p/{category}/{file_name[:-4]}.png'))

#%% ############################################################################
# Test on a subset
################################################################################
#%%
N = 20
# change path if necessary
path_dataset = Path('data/MIT_sampleVideos_RAW_final_25FPS_480x360p/')
# create random subset sample
subset = mifs.sample(n=N, random_state=2)

for index, row in subset.iterrows():
  category, file_name, best = row
  vr = VideoReader(str(path_dataset / category / file_name)) # AB: comment on VideoReader above
  
  best = mifs.loc[(mifs['category'] == category) & (mifs['fname'] == file_name)]['mif_idx'].values[0]
  
  mif = vr[best].asnumpy()
  print(mif.shape, best)

  #plt.imshow(mif)
  #plt.show()

  im = PIL.Image.fromarray(mif)
  # save sample both as PNG and JPG
  im.save(Path(f'data/MIFs/test/{category}_{file_name[:-4]}.png'))
  im.save(Path(f'data/MIFs/test/{category}_{file_name[:-4]}.jpeg'))


#%% ############################################################################
# Full Extraction 
################################################################################
#%%
import time, os
import PIL

start = time.time()

mif_format = 'png' # or 'png'
# change paths if necessary
path_dataset = Path('data/MIT_additionalVideos_25FPS_480x360p/')
path_output = Path(f'data/MIFs/MIT_additionalMIFs_480x360p_{mif_format}')

if not os.path.exists(path_output):
  os.mkdir(path_output)

i = 0
for index, row in mifs.iterrows():
  i+=1
  if i < len(mifs) + 1000:  # "+1000" to be sure :-P
    category, file_name, best = row
    
    path_output_category = path_output / category
    # Create output category directory if not present
    if not os.path.exists(path_output_category):
      os.mkdir(path_output_category)
        
    # Load video  
    vr = VideoReader(str(path_dataset / category / file_name))
    
    # MIF by idx
    best = mifs.loc[(mifs['category'] == category) & (mifs['fname'] == file_name)]['mif_idx'].values[0]
    mif = vr[best].asnumpy()

    # Save to image
    im = PIL.Image.fromarray(mif)
    im.save(path_output_category / f'{file_name[:-4]}.{mif_format}')
  
  else:
    break

stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/i:.3f}s per file)')

#%%
# Time expended
# JPEG: Time spent: 51.69s (~ 0.035s per file)
# PNG: Time spent: 102.16s (~ 0.070s per file)

#%%
# AB: What is this for?
path_mifs = Path('data/MIT_sampleVideos_RAW_final_25FPS_480x360p/mifs.csv')
mifs = pd.read_csv(path_mifs, usecols=['category', 'fname', 'mif_idx'])
len(mifs.category.unique())
# %%
