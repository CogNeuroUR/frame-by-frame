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
import shutil # AB: What for?
import glob

#%% Define input paths
# AB: Adjust these if neessary to personal location
path_gifs = Path('../input_data/TRIMMING/INPUT/MIT_GIFs_25FPS_480x360p_1.0s_TOP-3-PER-CAT_old+new')
path_pngs = Path('../input_data/TRIMMING/PNGs')
path_mp4s = Path('../input_data/TRIMMING/MP4s')

#%% Sweep through input gifs set
l_mp4s = []
l_missing = []
for path, subdirs, files in os.walk(path_mp4s):
  for name in files:
    if name[-3:] == 'mp4':
      l_mp4s.append([path.split('/')[-1],   # category
                       name])               # file name
    else:
      print('Ignored: ', name)
      
    if len(files) < 3: # AB: What does this accomplish?
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
# AB:  a bit more info abt this process please
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

#%% Convert to pandas.DataFrame
# Structure:
# Entry x: category_i, original_fname_x, renamed_x
df_lookup = pd.DataFrame(l_lookup,
                         columns=['category', 'fname', 'renamed'])

print(df_lookup.head()) # Feedback about structure

#%% Examples to access lookup dataframe
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

#%% Utility function
def check_mkdir(path):
  """
  Check if folder "path" exists. If not, creates one. 
  
  Returns
  -------
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
  Renames the input sets to a standardized lookup table.  # AB: all 3 of them?
  
  Parameters
  ------
  opt : dict
    Parameter dictionary
  df_lookup : pandas.DataFrame
  
  Returns
  -------
  True if no errors encountered.
  """
  
  # Check opt dictionary entries
  
  
  # Define format dictionary containing input and output paths per format
  # AB: Maybe some words considering Syntax or reference?
  dict_format = {'.mp4' : [opt['input_mp4s'], opt['output_path'] / 'MP4s'],
                 '.gif' : [opt['input_gifs'], opt['output_path'] / 'GIFs'],
                 '.png' : [opt['input_pngs'], opt['output_path'] / 'PNGs']}
  
  # Create output paths for each format
  check_mkdir(dict_format['.mp4'][1])
  check_mkdir(dict_format['.gif'][1])
  check_mkdir(dict_format['.png'][1])
  
  # Define (local) search function (inside renaming main function)
  def path_file(category, filename, format='.mp4'):
    """
    Looks for file of specific format in the given folder structure
    
    Parameters
    ----------
    category : str
    filename : str
    format : str
    
    Returns
    -------
    path : str
      Path to found file
    """
    if format == '.mp4':
      prefix = opt['input_mp4s']
    if format == '.gif':
      format = '*.gif'  # AB: Why does Format only switch here?
      prefix = opt['input_gifs']
    if format == '.png':
      prefix = opt['input_pngs']
    
    # Extract path with glob
    path_ = glob.glob(str(prefix / category / (filename + format)))
    
    # Check if single/multiple/none file/files were found 
    if len(path_) == 1:
      return path_[0] # "0" because it's a list of len=1
    else:
      raise Exception(f'glob.glob wasn\'t able to find ONLY a file at path: {path_[0]}')
    
  
  # Loop over df_lookup
  for i in range(len(df_lookup)):
    # Verbose
    if i % 100 == 0: print(f'\t{i}/{len(df_lookup)}')
    
    # Define category and file_name from lookup table
    category, file_name = df_lookup['category'][i], df_lookup['fname'][i]
    
    # Iterate over formats
    for format in ['.mp4', '.gif', '.png']:
      # Check if files exists per format and extract path to it
      path_to_file = path_file(category, file_name, format)
      # Create output directory
      check_mkdir(dict_format[format][1] / category)
      # Copy file to output path and rename it based on lookup table
      shutil.copy(path_to_file, dict_format[format][1] / category / (df_lookup['renamed'][i] + format))
      # AB:  What does shutil in detail?
  
  return True

#%% Run renaming
# Here the real renaming happens initiated above
# Path to output
# adjustpath if necessary
path_output = Path('input_data/RENAMING/')

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
if rename_lookup(opt, df_lookup):  # AB: Are all 3 formats renamed in this step?
  print('Succsseful!')

# Print elapsed time
stop = time.time()
duration = stop-start
print(f'\nTime elapsed: {duration:.2f}s (~ {duration/len(df_lookup):.2f}s per file)')

# %% Tests
# AB: Please elaborate, what exactly is tested here (only PNGs?)
# adjust if necessary
path_output = Path('../input_data/RENAMING/')

opt = {
  'input_mp4s' : path_mp4s,
  'input_gifs' : path_gifs,
  'input_pngs' : path_pngs,
  'output_path' : path_output,
}
format_ = 'png'
output = opt['output_path'] / 'PNGs' # or path_output_mp4s

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
