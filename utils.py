# Imports
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

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

