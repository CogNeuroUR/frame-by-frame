# Imports
import os
import sys
from pathlib import Path

# Video processing
if sys.version_info[1] != 8:
    import cv2
from moviepy.editor import ImageSequenceClip

import numpy as np
from scipy.cluster.hierarchy import dendrogram

from matplotlib import pyplot as plt
################################################################################
# Load categories (MiTv1)
################################################################################
def load_categories():
    path_prefix = Path().parent.absolute()
    """Load MiTv1 categories."""
    with open(path_prefix / 'labels/category_momentsv1.txt') as f:
        return [line.rstrip() for line in f.readlines()]


################################################################################
# topN_per_frame
################################################################################
def topN_per_frame(accuracies_dict, N=3,
                   extract_best_worst = False,
                   verbose = False):
    """
    Function to extract TopN accuracies per frame, given an accuracies dictionary.
    Required structure of the accuracies_dict is:
        {category_name : {video_fname : []}}
    """
    z = 1 # For verbose
    topN = {}
    
    # load categories
    l_categories = load_categories()
    
    if extract_best_worst:
        best_frame_dict = {}
        worst_frame_dict = {}
    
    # Iterate over categories in path_prefix
    for category_name in sorted(list(accuracies_dict.keys())):
        print(f'{category_name} {z}/{len(list(accuracies_dict.keys()))}')
        
        cat_idx = [i for i in range(len(l_categories)) 
                   if l_categories[i] == category_name][0]
        
        topN[category_name] = {}
        if extract_best_worst:
            best_frame_dict[category_name] = {}
            worst_frame_dict[category_name] = {}
        
        # Iterate over files in cateogory
        for video_fname in list(accuracies_dict[category_name].keys()):
            # Extract accuracies as (n_frames, 1) arrays
            per_frame_accuracies = np.array(accuracies_dict[category_name][video_fname])
            per_frame_accuracies.reshape((per_frame_accuracies.shape[0],
                                        len(list(accuracies_dict.keys()))))
            per_frame_accuracies = per_frame_accuracies[:, cat_idx]
            
            if False:
                print(f'\t{video_fname} : Max/Min accuracy at frame:' \
                f' {np.argmax(per_frame_accuracies)}/{np.argmin(per_frame_accuracies)}' \
                f' with value: {per_frame_accuracies[np.argmax(per_frame_accuracies)]}' \
                f' / {per_frame_accuracies[np.argmin(per_frame_accuracies)]}')

            if extract_best_worst:
                # Determined the index of the frame w/ max accuracy and write to dict
                best_frame_dict[category_name][video_fname] = (np.argmax(per_frame_accuracies),
                                                               per_frame_accuracies[np.argmax(per_frame_accuracies)])
                worst_frame_dict[category_name][video_fname] = (np.argmin(per_frame_accuracies),
                                                                per_frame_accuracies[np.argmin(per_frame_accuracies)])
                
            # Iterate over frames
            topN[category_name][video_fname] = {}
            for frame in range(len(accuracies_dict[category_name][video_fname])):

                frame_vals = accuracies_dict[category_name][video_fname][frame]
                
                indices = np.argsort(frame_vals)[::-1][:len(frame_vals)]
                values = np.array(frame_vals)[indices]
                
                topN[category_name][video_fname][frame] = []
                
                # Check if "category" is in TopN:
                if cat_idx in indices[:N]:
                    # l_categories[idx], sorted_acc[i]
                    #print(f'In Top{N}')
                    for i in range(len(indices[:N])):
                        idx = indices[i]
                        #print(idx, values[i])
                        topN[category_name][video_fname][frame].append([l_categories[idx],
                                                                        values[i]])
                else:
                    #print(f'Not in Top{N}')
                    for i in range(len(indices[:N-1])):
                        idx = indices[i]
                        #print(idx, values[i])
                        topN[category_name][video_fname][frame].append([l_categories[idx],
                                                                        values[i]])
                    #print(values[cat_idx])
                    
                    # Check the index here!!!
                    topN[category_name][video_fname][frame].append([l_categories[cat_idx],
                                                                    frame_vals[cat_idx]])   
        z += 1
    
    if extract_best_worst:
        return topN, best_frame_dict, worst_frame_dict
    else:
        return topN

################################################################################
# topN categories per file:
################################################################################
def topN_categories(accuracy_array, N=5,
                    make_plots=False,
                    verbose=False):
    """
    Function to extract TopN categories given accuracies of a file
    
    Returns the indexes of top N categories
    """
    # load categories
    labels = load_categories()
    
    # Extract the mean accuracies over frames
    mean_accuracies = np.mean(accuracy_array, axis=0)
    
    indices = np.argsort(mean_accuracies)[::-1][:len(mean_accuracies)]
    values = np.array(mean_accuracies)[indices]
    
    topN = []
    
    if verbose:
        print(f'--Top {N} Actions:')
    for i in range(N):
        if verbose:    
            print('{:.3f} -> {}'.format(values[i], labels[indices[i]]))
        topN.append(indices[i])
    
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
def topN_per_file(accuracies_dict, N=3,
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
    # load categories
    l_categories = load_categories()
    
    if extract_best_worst:
        best_frame_dict = []
        worst_frame_dict = []
    
    cat_idx = [i for i in range(len(l_categories))
               if l_categories[i] == category_name][0]

    # Extract accuracies as (n_frames, 1) arrays
    per_frame_accuracies = np.array(accuracies_dict[category_name][video_fname])
    #per_frame_accuracies.reshape((per_frame_accuracies.shape[0],
    #                            len(list(accuracies_dict.keys()))))
    #per_frame_accuracies = per_frame_accuracies[:, cat_idx]
    
    ass = per_frame_accuracies.reshape((per_frame_accuracies.shape[0],
                                            len(list(accuracies_dict.keys()))))
    ass = ass[:, cat_idx]
    if verbose:
        print(f'\t{video_fname} : Max/Min accuracy at frame:' \
        f' {np.argmax(ass)}/{np.argmin(ass)}' \
        f' with value: {ass[np.argmax(ass)]}' \
        f' / {ass[np.argmin(ass)]}')

    if extract_best_worst:
        # Determined the index of the frame w/ max accuracy and write to dict
        best_frame = (np.argmax(ass),
                      ass[np.argmax(ass)])
        worst_frame = (np.argmin(ass),
                       ass[np.argmin(ass)])
    
    # Initialize empty list
    topN = []
    
    # Check if "category" is in TopN:
    if cat_idx in topN_categories(per_frame_accuracies, N=N):
        indices = topN_categories(per_frame_accuracies, N=N)
        # Iterate over frames
        for frame in per_frame_accuracies:
            temp_list = []
            # Extract per-frame accuracy values:
            for idx in indices:
                #print(idx, values[i])
                temp_list.append(frame[ idx])
            # Append extracted to topN:
            topN.append(temp_list)
    else:
        indices = topN_categories(per_frame_accuracies, N=N-1)
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
def best_worst(accuracies_dict,
               category_name='applauding',
               video_fname='yt-bX3KmVN89Co_1.mp4',
               verbose = False):
    """
    Function to extract best and worst frames per file, given an accuracies dictionary,
    the name of the category and file to be examined:
    Required structure of the accuracies_dict is:
        {category_name : {video_fname : []}}
    """
    # load categories
    l_categories = load_categories()
    
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

################################################################################
# Dendrogram plotting (scikit-learn.org)
################################################################################
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
################################################################################
# GIF writting using moviepy
################################################################################

def gif(filename, array, fps=10, scale=1.0):
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
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    if scale != 1.0:
        clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    else:
        clip = ImageSequenceClip(list(array), fps=fps)
    clip.write_gif(filename, fps=fps, program='ffmpeg') #, program='ffmpeg') # or 'ImageMagick'
    return clip

#%% ############################################################################
# Three-frame differencing for motion detection
################################################################################
def outlaw_search(frames_array, threshold_value=25e6):
    """
    Function to search for "drastic" changes in a video segment based on "Frame
    Differencing. The outlaw criterion is based on the maximum value of the gradient of the
    frame-to-frame differences.
    
    Parameters:
    ---------------
    frames_array : ndarray (n_frames, height, width, n_channels)
        Array containing the frames of the video.
    threshold_value : int (optional)
        Value above which the video segment is subjected to exclusion.
    """
    # Compute three-frame differencing along the frames
    l_diffs_sum = []
    l_diffs = []
    for j in range(len(frames_array)):
        if j > 0:
            frame1 = frames_array[j-1]
            frame2 = frames_array[j]
            diffs = cv2.absdiff(frame1, frame2)
            l_diffs_sum.append(np.sum(diffs))
            l_diffs.append(diffs)
    # Check if differences are bigger than threshold
    if np.absolute(np.gradient(l_diffs_sum)).max() > threshold_value:
        return True
    else:
        return False

#%% ############################################################################
# Distance matrix computation (required for sklearn's AggClust)
################################################################################
def get_distances(X,model,mode='l2'):
    distances = []
    weights = []
    children=model.children_
    dims = (X.shape[1],1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
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
        weights.append( wNew)
    return distances, weights