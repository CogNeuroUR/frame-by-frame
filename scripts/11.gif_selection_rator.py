# 16.04.21 OV
#
# Extracts distribution of GIFs per MIF position for each participant in the
# GIF selection task using Rator distribution

#%% Imports
import os
from pathlib import Path

#%% Define input paths
# change path if needed
path_gifs = Path('../data/TRIMMING/INPUT/MIT_GIFs_25FPS_480x360p_1.0s_TOP-3-PER-CAT_old+new')

#%% Sweep through input gifs set # AB: why?
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

#%% Load Rator distribution
import pandas as pd
# rator distribution stored in .csv-file (displays which rator rated which categories)
path_sheet = Path('../saved/20210305_AB_ratorDistribution_qq1+qq2.csv')
df_rator = pd.read_csv(path_sheet,
                       skiprows=[0],
                       names=['LK', 'PS', 'LS', 'JS'],
                       delimiter=';',
                       usecols=[0, 1, 2, 3],
                      #  nan) # AB: this nan throws an error? remove?
                       )

print(df_rator)

#%%
# AB: What happens here? Please comment.
dict_participants = {}
for rater in list(df_rator.columns):
  l_b = []
  l_m = []
  l_e = []
  for category in df_rator[rater].dropna():
    #print(rater, category)
    path_cat = path_gifs / category
    l_files = [x.stem for x in path_cat.glob('**/*') if x.suffix == '.gif']

    # check file name extensions (indicate position)
    for file_name in l_files:
      if file_name[-2:] == '_b':
        l_b.append(file_name)
      if file_name[-2:] == '_m':
        l_m.append(file_name)
      if file_name[-2:] == '_e':
        l_e.append(file_name)
      if file_name[-2:] not in ['_b', '_m', '_e']:  
        print('File without MIF suffix: ', file_name)
  dict_participants[rater] = [len(l_b), len(l_m), len(l_e)]#[len(l_b), len(l_m), len(l_e)]

#%% Check
flat_list = [item for sublist in list(dict_participants.values()) for item in sublist]

assert sum(flat_list) == len(l_gifs) # check for same length

# %% Plot distribution
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

temp = []
names = ['Lisa', 'Lucca', 'Pauline', 'Johannes']
positions = ['B', 'M', 'E']

for position in range(3):
  for participant in list(df_rator.columns):
    temp.append([positions[position], participant,
                 dict_participants[participant][position]])

df_distrib = pd.DataFrame(temp, columns=['Position', 'Participants', 'Count'])

print(df_distrib)

# Check if all files visited:
assert df_distrib['Count'].sum() == len(l_gifs), "Not all files visited!"

#%% Plot
sns.catplot(data = df_distrib,
            x = 'Position',
            y = 'Count',
            hue = 'Participants',
            kind='bar')
# uncomment for savign plot (pay attention to output path)
#plt.savefig('plots/gif_distribution_per_participant.pdf')
