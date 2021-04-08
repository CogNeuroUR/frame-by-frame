# 24.02.21 OV
# Distribution
################################################################################
# Imports
################################################################################
#%% Imports
import os
from pathlib import Path

#%% Sweep through input gifs set
l_b = []
l_m = []
l_e = []
path_gifs = Path('data/GIFs/BW_GIFs')

for path, subdirs, files in os.walk(path_gifs):
  for name in files:
    if name[-3:] == 'gif':
      if name[-6:-4] == '_b':
        l_b.append(name)
      if name[-6:-4] == '_m':
        l_m.append(name)
      if name[-6:-4] == '_e':
        l_e.append(name)
      
print(len(l_b), len(l_m), len(l_e))


#%% ############################################################################
# Extract from list of videos
################################################################################
import time, os
import shutil

path_output_pngs = path_output / 'PNGs'
path_output_mp4s = path_output / 'MP4s'

check_mkdir(path_output)
check_mkdir(path_output_pngs)  
check_mkdir(path_output_mp4s)
# ==========================================================

start = time.time()

for i in range(len(l_gifs)):
  category, file_name = l_gifs[i]
  
  # Trim the file format and the MIF position identifier, e.g. '_b' 
  trimmed_fname = file_name[:-6]
  suffix = Path(category) / trimmed_fname
  
  # Verbose
  if i % 50 == 0:
    print(f'\n{i}/{len(l_gifs)}')

  # Create "category" folder if necessary
  check_mkdir(path_output_pngs / category)
  check_mkdir(path_output_mp4s / category)
  
  # Copy files in output folders
  dst_png = (path_output_pngs / suffix).with_suffix('.png')
  dst_mp4 = (path_output_mp4s / suffix).with_suffix('.mp4')
  shutil.copyfile((path_pngs / suffix).with_suffix('.png'), dst_png)
  shutil.copyfile((path_mp4s / suffix).with_suffix('.mp4'), dst_mp4)
    
stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/i:.2f}s per file)')

# %% Tests
output = path_output_mp4s # or path_output_pngs
format_ = 'mp4'
l_test = []

for path, subdirs, files in os.walk(path_output_mp4s):
  for name in files:
    if name[-3:] == format_:
      l_test.append([path.split('/')[-1],   # category
                       name])                 # file name
    else:
      print('Ignored: ', name)

print(f'Total nr. of {format_}-s in {path_output_mp4s}:', len(l_test))
# %%
