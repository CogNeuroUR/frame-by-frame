# [08.03.21] OV
# Script for renaming MIF, i.e. PNG, and MP4 datasets using the input GIF dataset
#
# Idea (OV):
# Having the original naming of the files in the gif, png and mp4 versions:
# create lookup table for a further renaming of the files in the format:
# [category_i, filename_j[:-4], category_i_k], (k : {1, 2, 3})

#%% Imports
import os
from pathlib import Path
import pandas as pd
import time
import shutil # offers high-level operations on files, like copying
import glob

#%% Import custom utils
import sys, importlib
sys.path.insert(0, '..')
import utils
importlib.reload(utils) # Reload If modified during runtime


#%% Define input paths
# AB: Adjust these if neessary to personal location
path_gifs = Path('../input_data/TRIMMING/INPUT/MIT_GIFs_25FPS_480x360p_1.0s_TOP-3-PER-CAT_old+new')
path_pngs = Path('../input_data/TRIMMING/PNGs')
path_mp4s = Path('../input_data/TRIMMING/MP4s')

#%% Get a list of MP4s based on input GIFs
# Empty lists for collecting mp4s and categories w/ missing files,
# i.e. w/ less than three files 
l_mp4s = []
l_missing = []

# Sweep through path to mp4s
for path, subdirs, files in os.walk(path_mp4s):
  for name in files:
    # Check if file has required format
    if name[-3:] == 'mp4':
      l_mp4s.append([path.split('/')[-1],   # category
                       name])               # file name
    else:
      print('Ignored: ', name)
      
    # Check if there are less than three files per category
    if len(files) < 3:
      l_missing.append([path.split('/')[-1], len(files)])

if l_missing:
  l_missing.pop(0)

if l_mp4s:
  l_mp4s = sorted(l_mp4s)
print('Total nr. of files: ', len(l_mp4s))

#%%#############################################################################
# Create lookup table
################################################################################
#%% Create lookup list based on l_mp4s
# Aimed structure:
# [
# [category_i, 'original_fname_1', 'category_i_1_],
# [category_i, 'original_fname_2', 'category_i_2],
# [category_i, 'original_fname_3', 'category_i_3],
# [category_j, 'original_fname_1', 'category_j_1],
# ...
# ]

l_lookup = []

# Initialize file index
k = 1
for i in range(len(l_mp4s)):
  category, file_name = l_mp4s[i]
  
  if i > 0:
    # Check if i-th entry is equal to (i-1)th entry
    if category == l_mp4s[i-1][0]:
      k += 1 # increment
    # Otherwise, renew to 1
    else:
      k = 1

  # Append [category, original_fname, new_fname]
  l_lookup.append([category, file_name[:-4], category + '_' + str(k)])

#%% Convert to pandas.DataFrame
# Structure:
# Entry x: category_i, original_fname_x, renamed_x
df_lookup = pd.DataFrame(l_lookup,
                         columns=['category', 'fname', 'renamed'])

print(df_lookup.head()) # Feedback about structure

#%% Examples of accessing lookup dataframe
x = 234  # example
print(f'{x}-th entry has category: ')
print(df_lookup['category'][x])

print(f'\n{x}-th entry has original fname: ')
print(df_lookup['fname'][x])

print(f'\n{x}-th entry will be renamed to: ')
print(df_lookup['renamed'][x])

#%% Save as csv
df_lookup.to_csv('input_data/RENAMING/lookup_table.csv')

#%%#############################################################################
# Renaming
################################################################################
#%% Tests with glob
path_2_file = path_gifs / df_lookup['category'][i] / (df_lookup['fname'][i] + '*.gif')
print(len(glob.glob(str(path_2_file))),
          glob.glob(str(path_2_file)))

#%% Run renaming for all the three formats
# Here the real renaming happens initiated above
# Path to output (adjustpath if necessary)
path_output = Path('../input_data/RENAMING/')

# Parameter dictionary for rename_lookup() function
opt = {
  'input_mp4s' : path_mp4s,
  'input_gifs' : path_gifs,
  'input_pngs' : path_pngs,
  'output_path' : path_output,
}

# For elapsed time
start = time.time()

# Run renaming
if utils.rename_lookup(opt, df_lookup):
  print('Succsseful!')

# Print elapsed time
stop = time.time()
duration = stop-start
print(f'\nTime elapsed: {duration:.2f}s (~ {duration/len(df_lookup):.2f}s per file)')

# %% Test if files (e.g. png-s) were written to selected output directory 
# adjust if necessary
path_output = Path('../input_data/RENAMING/')

opt = {
  'input_mp4s' : path_mp4s,
  'input_gifs' : path_gifs,
  'input_pngs' : path_pngs,
  'output_path' : path_output,
}
format_ = 'png' # or 'mp4'; or 'gif'
output = opt['output_path'] / 'PNGs' # or 'MP4s'; or 'GIFs'

l_test = []

for path, subdirs, files in os.walk(output):
  for name in files:
    if name[-3:] == format_:
      l_test.append([path.split('/')[-1],   # category
                       name])               # file name
    else:
      print('Ignored: ', name)

print(f'Total nr. of {format_}-s in {output}:', len(l_test))

# %%
