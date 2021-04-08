# [05.03.21] OV
# Trimming the PNG and MP4 sets based on final version of GIF set

################################################################################
# Imports
################################################################################
#%% Imports
import os
from pathlib import Path
import pandas as pd
import time
import shutil
import glob

#%% Define input paths
path_gifs = Path('data/TRIMMING/INPUT/MIT_GIFs_25FPS_480x360p_1.0s_TOP-3-PER-CAT_old+new')
path_pngs = Path('data/TRIMMING/PNGs')
path_mp4s = Path('data/TRIMMING/MP4s')

#%% Define task
# Having the original naming of the files in the gif, png and mp4 versions:
# create lookup table for a further renaming of the files in the format:
# [category_i, filename_j[:-4], category_i_k], (k : {1, 2, 3})

#%% Sweep through input gifs set
l_mp4s = []
l_missing = []
for path, subdirs, files in os.walk(path_mp4s):
  for name in files:
    if name[-3:] == 'mp4':
      l_mp4s.append([path.split('/')[-1],   # category
                       name])                 # file name
    else:
      print('Ignored: ', name)
      
    if len(files) < 3:
      l_missing.append([path.split('/')[-1], len(files)])

l_missing.pop(0)

if l_mp4s:
  l_mp4s = sorted(l_mp4s)
print('Total nr. of files: ', len(l_mp4s))

#%% Create lookup table as list based on l_mp4s
l_lookup = []

k = 1
for i in range(len(l_mp4s)):
  category, file_name = l_mp4s[i]
  
  if i > 0:
    if category == l_mp4s[i-1][0]:
      k += 1
 
    else:
      k = 1

  l_lookup.append([category, file_name[:-4], category + '_' + str(k)])

#%% To DataFrame
df_lookup = pd.DataFrame(l_lookup,
                         columns=['category', 'fname', 'renamed'])

print(df_lookup.head())

#%% Save as csv
df_lookup.to_csv('data/RENAMING/lookup_table.csv')

#%%#############################################################################
# Renaming
################################################################################
#%% Tests with glob
path_2_file = path_gifs / df_lookup['category'][i] / (df_lookup['fname'][i] + '*.gif')
print(len(glob.glob(str(path_2_file))),
          glob.glob(str(path_2_file)))

#%% Utility function
def check_mkdir(path):
  """
  Check if folder "path" exists.
  If exists, returns "True", otherwise create a folder at "path" and return "False"
  """
  if not os.path.exists(path):
    os.mkdir(path)
    return False
  else:
    return True

#%% Main renaming function
def rename_lookup(opt, df_lookup):
  """
  Renames the input sets to a standardized lookup table.
  
  Params
  ======
  opt : dict
    Parameter dictionary
  df_lookup : pandas.DataFrame
  """
  # Check opt dictionary entries
  # ...
  
  # 
  dict_format = {'.mp4' : [opt['input_mp4s'], opt['output_path'] / 'MP4s'],
                 '.gif' : [opt['input_gifs'], opt['output_path'] / 'GIFs'],
                 '.png' : [opt['input_pngs'], opt['output_path'] / 'PNGs']}
  
  # Create output paths for each format
  check_mkdir(dict_format['.mp4'][1])
  check_mkdir(dict_format['.gif'][1])
  check_mkdir(dict_format['.png'][1])
  
  def path_file(category, filename, format='.mp4'):
    if format == '.mp4':
      prefix = opt['input_mp4s']
    if format == '.gif':
      format = '*.gif'
      prefix = opt['input_gifs']
    if format == '.png':
      prefix = opt['input_pngs']
    
    # Extract path with glob
    path_ = glob.glob(str(prefix / category / (filename + format)))
    if len(path_) == 1:
      return path_[0]
    else:
      raise Exception(f'glob.glob wasn\' able to find ONLY a file at path: {path_[0]}')
    
  
  # Loop over df_lookup
  for i in range(len(df_lookup)):
    if i % 100 == 0: print(f'\t{i}/{len(df_lookup)}')
    category, file_name = df_lookup['category'][i], df_lookup['fname'][i]
    
    # Check if files exists per format and extract path to it
    for format in ['.mp4', '.gif', '.png']:
      path_to_file = path_file(category, file_name, format)
      
      # Copy file to output path and rename it based on lookup table
      check_mkdir(dict_format[format][1] / category)
      shutil.copy(path_to_file, dict_format[format][1] / category / (df_lookup['renamed'][i] + format))
  
  return True

#%% Run renaming
path_output = Path('data/RENAMING/')

opt = {
  'input_mp4s' : path_mp4s,
  'input_gifs' : path_gifs,
  'input_pngs' : path_pngs,
  'output_path' : path_output,
}

start = time.time()
if rename_lookup(opt, df_lookup):
  print('Succsseful!')
stop = time.time()
duration = stop-start
print(f'\nTime elapsed: {duration:.2f}s (~ {duration/len(df_lookup):.2f}s per file)')

# %% Tests
format_ = 'png'
output = opt['output_path'] / 'PNGs' # or path_output_mp4s
l_test = []

for path, subdirs, files in os.walk(output):
  for name in files:
    if name[-3:] == format_:
      l_test.append([path.split('/')[-1],   # category
                       name])                 # file name
    else:
      print('Ignored: ', name)

print(f'Total nr. of {format_}-s in {output}:', len(l_test))

