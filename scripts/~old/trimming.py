# [24.02.21] OV
# Trimming the PNG and MP4 sets based on final version of GIF set

################################################################################
# Imports
################################################################################
#%% Imports
import os
from pathlib import Path

#%% Define input paths
path_gifs = Path('input_data/TRIMMING/INPUT/MIT_GIFs_25FPS_480x360p_1.0s_TOP-3-PER-CAT_old+new')
path_pngs = Path('input_data/TRIMMING/INPUT/MIT_MIFs_480x360p_png_old+new')
path_mp4s = Path('input_data/TRIMMING/INPUT/08_MIT_sampleVideos_25FPS_480x360p_old+new')
path_output = Path('input_data/TRIMMING/')

#%% Define task
# Having final GIFs in path_gifs, sweep through categories -> fnames
# * extract fnames and remove suffixes, e.g. '_b.mp4'
# * use trimmed fname to copy mp4s and pngs to an output directory

#%% Sweep through input gifs set
l_gifs = []
l_missing = []
for path, subdirs, files in os.walk(path_gifs):
  for name in files:
    if name[-3:] == 'gif':
      l_gifs.append([path.split('/')[-1],   # category
                       name])                 # file name
    else:
      print('Ignored: ', name)
      
    if len(files) < 3:
      l_missing.append([path.split('/')[-1], len(files)])

l_missing.pop(0)

if l_gifs:
  l_gifs = sorted(l_gifs)
print('Total nr. of GIFs: ', len(l_gifs))

#%% Define useful function
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

###########################
# Distribution parameters
p = 0 # for participant ID : Lisa (Lisa, Lucca, Pauline, Johannes)
l_b = []
l_m = []
l_e = []
dict_participants = {}

###########################
trimming = True
###########################
diving_run = False
packaging_run = False
smiling_run = False

for i in range(len(l_gifs)):
  category, file_name = l_gifs[i]
  
  # Add new entry to participants dictionary if 
  has_run = False
  if category == 'diving' and diving_run == False: # Lucca
    dict_participants[p] = [len(l_b), len(l_m), len(l_e)] # [l_b, l_m, l_e]
    p = 1
    l_b = []
    l_m = []
    l_e = []
    diving_run = True
    
  if category == 'packaging' and packaging_run == False: # Pauline
    dict_participants[p] = [len(l_b), len(l_m), len(l_e)] #[l_b, l_m, l_e]
    p = 2
    l_b = []
    l_m = []
    l_e = []
    packaging_run = True

  if category == 'smiling' and smiling_run == False: # Johannes
    dict_participants[p] = [len(l_b), len(l_m), len(l_e)]
    p = 3
    l_b = []
    l_m = []
    l_e = []
    smiling_run = True
  
  if file_name[-6:-4] == '_b':
    l_b.append(file_name)
  if file_name[-6:-4] == '_m':
    l_m.append(file_name)
  if file_name[-6:-4] == '_e':
    l_e.append(file_name)
  if file_name[-6:-4] not in ['_b', '_m', '_e']:
    print('File without MIF suffix: ', file_name)
  
  # Perform "trimming" of the file names
  if trimming:
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

dict_participants[p] = [len(l_b), len(l_m), len(l_e)]

stop = time.time()
duration = stop-start
print(f'\nTime spent: {duration:.2f}s (~ {duration/i:.2f}s per file)')

# %% Tests
format_ = 'png'
output = path_output_pngs # or path_output_mp4s
l_test = []

for path, subdirs, files in os.walk(output):
  for name in files:
    if name[-3:] == format_:
      l_test.append([path.split('/')[-1],   # category
                       name])                 # file name
    else:
      print('Ignored: ', name)

print(f'Total nr. of {format_}-s in {output}:', len(l_test))

# %% Plot distribution
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

temp = []
names = ['Lisa', 'Lucca', 'Pauline', 'Johannes']
positions = ['B', 'M', 'E']

for position in range(3):
  for participant in range(4):
    temp.append([positions[position], names[participant],
                 dict_participants[participant][position]])

df_distrib = pd.DataFrame(temp, columns=['Position', 'Participants', 'Count'])

print(df_distrib.head())

# Check if all files visited:
assert df_distrib['Count'].sum() == len(l_gifs), "Not all files visited!"


sns.catplot(data = df_distrib,
            x = 'Position',
            y = 'Count',
            hue = 'Participants',
            kind='bar')
#plt.savefig('plots/gif_distribution_per_participant.pdf')

# %% Tests
x = df_distrib['Count']
input_list = [l_gifs[i][1] for i in range(len(l_gifs))]
flat_list = [item for sublist in x.tolist() for item in sublist]

print(list(set(input_list) - set(flat_list)))

# %%
i = 0
for i in range(len(l_gifs)):
  if len(l_gifs[i]) < 3:
    i += 1
    #print(l_gifs[i][0])
    
print(i)

# %%
import pandas as pd
l_missing.sort()
missing = pd.DataFrame(l_missing, columns=['Category', 'Count'])
print(missing)

missing.to_csv('spreadsheets/lacking_categories_02.03.21.csv')
# %%
