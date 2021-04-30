# Imports
import os
import sys
import time
import numpy as np

# Video processing
from moviepy.editor import ImageSequenceClip
if sys.version_info[1] != 8:
    import cv2
    
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

################################################################################
# Load categories (MiTv1)
################################################################################
def load_categories(path_to_categories):
    """Load MiTv1 categories given path to .txt file."""
    with open(path_to_categories) as f:
        return [line.rstrip() for line in f.readlines()]


################################################################################
# topN categories per file:
################################################################################
def topN_categories(accuracy_array, l_categories,
                    N=5,
                    make_plots=False,
                    verbose=False):
  """
  Extracts TopN categories' indices given softmax ccuracies for a file
  
  Parameters
  ----------
  accuracy_array : numpy.ndarray
    Accuracy array
  l_categories : list
    Labels used to train the model.
  N : int
    TopN categories to inspect. Default: 5
  make_plots : bool
    If to make accuracy plots. Default: False
  verbose : bool
    For user feedback. Default: False

  Returns
  -------
  topN : list
    Indices of topN categories.
  """
  
  # Extract the mean accuracies over frames
  mean_accuracies = np.mean(accuracy_array, axis=0)
  
  # Collect ordered indices & acc. values
  indices = np.argsort(mean_accuracies)[::-1][:len(mean_accuracies)]
  values = np.array(mean_accuracies)[indices]
  
  topN = []
  
  # User feedback
  if verbose:
    print(f'--Top {N} Actions:')
  for i in range(N):
    if verbose:    
      print('{:.3f} -> {}'.format(values[i], l_categories[indices[i]]))
    topN.append(indices[i])

  # Make plots
  if make_plots:
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(mean_accuracies)),
           height = mean_accuracies)
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
          label.set_visible(True)
        else:
          label.set_visible(False)
    ax.set_title(f'Prediction accuracty')
    ax.set_xlabel('Frame nr.')
    ax.set_ylabel('Prediction accuracy (softmax)')
    plt.show()

  return topN


################################################################################
# topN_per_file
################################################################################
def topN_per_file(accuracies_dict, l_categories,
                  N=3,
                  category_name='applauding',
                  video_fname='yt-bX3KmVN89Co_1.mp4',
                  extract_best_worst = False,
                  verbose = False):
  """
  Function to extract TopN accuracies per frame, given an accuracies dictionary,
  the name of the category and file to be examined:
  Required structure of the accuracies_dict is:
      {category_name : {video_fname : []}}
  """

  if extract_best_worst:
    best_frame_dict = []
    worst_frame_dict = []
  
  # Extract index of given category (0-338)
  cat_idx = [i for i in range(len(l_categories))
              if l_categories[i] == category_name][0]

  # Extract accuracies for all categories for given file
  per_frame_accuracies = np.array(accuracies_dict[category_name][video_fname])
  
  # Reshape to (n_frames, n_labels)
  ass = per_frame_accuracies.reshape((per_frame_accuracies.shape[0],
                                      len(list(accuracies_dict.keys()))))
  # Extract values only from TRUE label: (n_frames, 1)
  ass = ass[:, cat_idx]

  # For user feedback
  if verbose:
    print(f'\t{video_fname} : Max/Min accuracy at frame:' \
    f' {np.argmax(ass)}/{np.argmin(ass)}' \
    f' with value: {ass[np.argmax(ass)]}' \
    f' / {ass[np.argmin(ass)]}')

  # Check if to additionally extract best and worst frame idx of TRUE category
  if extract_best_worst:
    # Determined the index of the frame w/ max accuracy and write to dict
    best_frame = (np.argmax(ass),
                  ass[np.argmax(ass)])
    worst_frame = (np.argmin(ass),
                    ass[np.argmin(ass)])

  # Initialize list for per-frame accuracy values of the topN categories
  topN = []
  
  # Check if "category" is in TopN:
  if cat_idx in topN_categories(per_frame_accuracies, l_categories=l_categories,N=N):
    indices = topN_categories(per_frame_accuracies, l_categories=l_categories,N=N)
    # Iterate over frames
    for frame in per_frame_accuracies:
      temp_list = []
      # Extract per-frame accuracy values:
      for idx in indices:
          #print(idx, values[i])
          temp_list.append(frame[idx])
      # Append extracted to topN:
      topN.append(temp_list)
  else:
    # Extract values only for topN-1 categories and append values for TRUE one
    indices = topN_categories(per_frame_accuracies, l_categories=l_categories,N=N-1)
    indices.append(cat_idx)
    
    # Iterate over frames
    for frame in per_frame_accuracies:
      temp_list = []
      # Extract per-frame accuracy values:
      for idx in indices:
          temp_list.append(frame[idx])
      # Append extracted to topN:
      topN.append(temp_list)
      
  if extract_best_worst:
    return np.array(l_categories)[indices], np.array(topN).T, best_frame, worst_frame
  else:
    return np.array(l_categories)[indices], np.array(topN).T


################################################################################
# best_worst
################################################################################
def best_worst(accuracies_dict, l_categories,
               category_name='applauding',
               video_fname='yt-bX3KmVN89Co_1.mp4',
               verbose = False):
  """
  Extracts best and worst frames per file, given an accuracies dictionary,
  the name of the category and file to be examined.

  Parameters
  ----------
  accuracies_dict : dict
    Dictionary of accuracies of structure
    {category_name : {video_fname : []}}

  l_categories : list
    Labels used to train the model.
  category_name : str
    Category name. Default: 'applauding'
  video_fname : str
    Video filename. Default: 'yt-bX3KmVN89Co_1.mp4'
  verbose : bool
    For user feedback. Default: False

  Returns
  -------
  best_frame : tuple
    (Best frame idx, accuracy value)
  worst_frame) : tuple
    (Worst frame idx, accuracy value)
  """
  cat_idx = [i for i in range(len(l_categories))
              if l_categories[i] == category_name][0]

  # Extract accuracies as (n_frames, 1) arrays
  per_frame_accuracies = np.array(accuracies_dict[category_name][video_fname])

  ass = per_frame_accuracies.reshape((per_frame_accuracies.shape[0],
                                          len(list(accuracies_dict.keys()))))
  ass = ass[:, cat_idx]
  if verbose:
      print(f'\t{video_fname} : Max/Min accuracy at frame:' \
      f' {np.argmax(ass)}/{np.argmin(ass)}' \
      f' with value: {ass[np.argmax(ass)]}' \
      f' / {ass[np.argmin(ass)]}')

  # Determined the index of the frame w/ max accuracy and write to dict
  best_frame = (np.argmax(ass), ass[np.argmax(ass)])
  worst_frame = (np.argmin(ass), ass[np.argmin(ass)])
  
  return best_frame, worst_frame


#%% ############################################################################
# extract_features
################################################################################
def extract_features(l_categories, softmax_dict):
  """
  Extracts softmax feature vectors, given an accuracies dictionary.
  The accuracies dictionary of dictionaries contains per category, per file,
  per frame softmax classification probabilities for the true category.

  Parameters
  ----------
  l_categories : list
    Original labels used for pretraining of the model (ordered as for the model pretraining)
  
  softmax_dict : dict
    Accuracies dictionary of following structure:
    {
      category_i :
        {
          file_name_j : (n_frames, n_labels)
        }
    }
      
  Outputs
  -------
  features_softmax : array-like, shape (n_samples, n_features)
    Softmax feature vectors
  """
  # Idea:
  # Iterate over categories in l_categories

  # Define empty lists for collecting features vectors & labels
  l_avg_softvectors = []
  l_labels = []

  start = time.time()   # For elapsed time

  for category in l_categories:
    
    # Extract filenames per category
    l_files = list(softmax_dict[category].keys())
    
    if not l_files: # Check if any files present in current category
      continue
    
    # Iterate over files & collect avg accuracies on the TRUE category per file
    l_softmax = [] # list to collect raw feature vectors
    for file_name in l_files:
      # Determine the best (and worst) frame idx.s
      best, worst = best_worst(accuracies_dict=softmax_dict,
                               l_categories=l_categories,
                               category_name=category,
                               video_fname=file_name)
      
      # Read softmax vector at best frame -> (n_categories, 1)
      per_frame_accuracies = np.array(
        softmax_dict[category][file_name])[best[0]]
      # Append into raw list
      l_softmax.append(per_frame_accuracies)

    # Convert l_softmax to array:
    l_softmax = np.array(l_softmax)
    
    # Append the mean softmax vector per category into l_avg_softvectors
    l_avg_softvectors.append(l_softmax.mean(axis=0))
    # Append TRUE category
    l_labels.append(str(category))

  # Convert l_avg_softvectors to an array of softmax features
  features_softmax = np.array(l_avg_softvectors)
  
  # For elapsed time
  stop = time.time()
  duration = stop-start
  print(f'Time spent: {duration:.2f}s (~ {duration/len(l_categories):.4f}s per file)')
  print(f'{len(l_avg_softvectors)} categories swept!')

  return features_softmax, l_labels
    

################################################################################
# GIF writting using moviepy
################################################################################
def gif(filename, array, fps=10, scale=1.0, scale_width=None):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : numpy.ndarray
        Array with sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    scale_width : int
      Scalling factor (default: None)
    
    Returns
    -------
    clip : moviepy.video.VideoClip.VideoClip
      Video clip from given sequence of frames
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    if scale != 1.0 and scale_width==None:
        clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    elif scale_width != None:
            clip = ImageSequenceClip(list(array), fps=fps).resize(width=scale_width)
    else:
        clip = ImageSequenceClip(list(array), fps=fps)
    #clip.write_gif(filename, fps=fps, program='ffmpeg') #, program='ffmpeg') # or 'ImageMagick'
    clip.write_gif(filename, fps=fps, progress_bar=False)
    return clip


#%% ############################################################################
# Three-frame differencing for motion detection
################################################################################
def outlaw_search(frames_array, threshold_value=25e6):
    """
    Search for "drastic" changes in a video segment based on "Frame
    Differencing. The outlaw criterion is based on the maximum value of the gradient of the
    frame-to-frame differences.
    
    Parameters:
    ---------------
    frames_array : ndarray (n_frames, height, width, n_channels)
        Array containing the frames of the video.
    threshold_value : int (optional)
        Value above which the video segment is subjected to exclusion.

    Returns
    -------
    True if criterion met, otherwise False.
    """

    # Empty lists for collecting per-frame differences and their sum
    l_diffs_sum = []
    l_diffs = []

    # Compute two-frame differencing along the frames
    for j in range(len(frames_array)):
        if j > 0:
            frame1 = frames_array[j-1]
            frame2 = frames_array[j]
            diffs = cv2.absdiff(frame1, frame2)
            l_diffs_sum.append(np.sum(diffs))
            l_diffs.append(diffs)

    # Check if differences are bigger than threshold
    # i.e. take the maximum value from the gradient of summed differences
    if np.absolute(np.gradient(l_diffs_sum)).max() > threshold_value:
        return True
    else:
        return False

#%% ############################################################################
# Distance matrix computation (required for sklearn's AggClust)
################################################################################
def get_distances(X,model,mode='l2'):
  """
  Computes weights and distances to be passed to sklearn's dendrogram()

  Parameters
  ----------
  X : numpy.ndarray
    Features/Representational dissimilarity matrix
  model : sklearn.cluster.AgglomerativeClustering
    Fitted clustering model
  mode : str
    How to deal with a higher level cluster merge with lower distance.
    (Default: 'l2')

  From https://stackoverflow.com/questions/26851553/sklearn-agglomerative-clustering-linkage-matrix/47769506#47769506
  """

  # Empty distance and weight lists
  distances = []
  weights = []

  # Get children of each non-leaf node from model
  children=model.children_
  
  # Dimensions to reshape 
  dims = (X.shape[1],1)

  # Temporary dicts
  distCache = {}
  weightCache = {}

  # Iterate over children
  for childs in children:
    # Reshape samples
    c1 = X[childs[0]].reshape(dims)
    c2 = X[childs[1]].reshape(dims)
    
    # Initiate distances & weights
    c1Dist = 0
    c1W = 1
    c2Dist = 0
    c2W = 1
    
    if childs[0] in distCache.keys():
      c1Dist = distCache[childs[0]]
      c1W = weightCache[childs[0]]
    if childs[1] in distCache.keys():
      c2Dist = distCache[childs[1]]
      c2W = weightCache[childs[1]]
    d = np.linalg.norm(c1-c2)
    cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)

    X = np.vstack((X,cc.T))

    newChild_id = X.shape[0]-1

    # How to deal with a higher level cluster merge with lower distance:
    if mode=='l2':  # Increase the higher level cluster size suing an l2 norm
      added_dist = (c1Dist**2+c2Dist**2)**0.5 
      dNew = (d**2 + added_dist**2)**0.5
    elif mode == 'max':  # If the previrous clusters had higher distance, use that one
      dNew = max(d,c1Dist,c2Dist)
    elif mode == 'actual':  # Plot the actual distance.
      dNew = d

    wNew = (c1W + c2W)
    distCache[newChild_id] = dNew
    weightCache[newChild_id] = wNew

    distances.append(dNew)
    weights.append(wNew)
  return distances, weights

################################################################################
# Check for path 
################################################################################

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