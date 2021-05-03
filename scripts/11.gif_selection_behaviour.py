# 16.04.21 OV
#
# Extracts distribution of GIFs per MIF position for each participant in the
# GIF selection task

#%% Imports
import os
from pathlib import Path
import time

#%% Define input paths
# change paths if necessary
path_gifs = Path('../input_data/TRIMMING/INPUT/MIT_GIFs_25FPS_480x360p_1.0s_TOP-3-PER-CAT_old+new')

#%% Sweep through input gifs set
l_gifs = []
l_missing = []
for path, subdirs, files in os.walk(path_gifs):
  for name in files:
    if name[-3:] == 'gif':
      l_gifs.append([path.split('/')[-1],   # category
                       name])               # file name
    else:
      print('Ignored: ', name)
      
    if len(files) < 3:
      l_missing.append([path.split('/')[-1], len(files)])

l_missing.pop(0)

if l_gifs:
  l_gifs = sorted(l_gifs)
print('Total nr. of GIFs: ', len(l_gifs))

#%% ############################################################################
# Extract position of MIF in GIF (_b(eginning), _m(iddle), _e(nd)) from list of videos
################################################################################
# Distribution parameters
p = 0 # for participant ID : Lisa (Lisa, Lucca, Pauline, Johannes)
l_b = []
l_m = []
l_e = []
dict_participants = {}

###########################
diving_run = False
packaging_run = False
smiling_run = False

# AB: Please comment a bit on the logic of your loop here

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
  
  # AB: Also comment here a bit on the logik.
  # I got it but would help speed up the understanding of the code in the future
  if file_name[-6:-4] == '_b':
    l_b.append(file_name)
  if file_name[-6:-4] == '_m':
    l_m.append(file_name)
  if file_name[-6:-4] == '_e':
    l_e.append(file_name)
  if file_name[-6:-4] not in ['_b', '_m', '_e']:
    print('File without MIF suffix: ', file_name)

dict_participants[p] = [len(l_b), len(l_m), len(l_e)]

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

#%% Plot
sns.catplot(data = df_distrib,
            x = 'Position',
            y = 'Count',
            hue = 'Participants',
            kind='bar')
#plt.savefig('plots/gif_distribution_per_participant.pdf')

# %% Tests
# AB: What exactly is tested here?
x = df_distrib['Count']
input_list = [l_gifs[i][1] for i in range(len(l_gifs))]
# flatten list
flat_list = [item for sublist in x.tolist() for item in sublist]

print(list(set(input_list) - set(flat_list)))
