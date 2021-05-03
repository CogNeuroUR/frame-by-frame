#%% 
import os
from pathlib import Path

path_gifs = Path('input_data/MIT_GIFs_25FPS_480x360p_1.0s/')

l_1 = 0
l_2 = 0
l_3 = 0
l_4 = 0
l_5 = 0
l_rest = 0

for category in os.listdir(path_gifs):
  if category != 'mifs.csv' and category != 'readme.txt':
    for file in os.listdir(path_gifs / category):  
      dlist = os.listdir(path_gifs / category / file)
      if len(dlist) == 1:
        l_1 += 1
      if len(dlist) == 2:
        l_2 += 1
      if len(dlist) == 3:
        l_3 += 1
      
mif_distr = [l_1, l_2, l_3]

import matplotlib.pyplot as plt

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

fig, ax = plt.subplots(figsize=(8,5))
rect = ax.bar(['1', '2', '3'], mif_distr)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('GIFs per video', fontsize=15)
ax.set_ylim((0, max(mif_distr) + 100))

autolabel(rect)
fig.tight_layout()
plt.savefig('plots/gif_distrib.png')
plt.show()

# %%
