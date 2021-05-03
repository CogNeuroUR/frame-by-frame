# MIT-Pipeline
[OV; AB

## Prerequirements / Initial Info

+ **platform independence**: tested under independently on CoLab, Mac and Linux = OK
+ **paths** are relative (clone GitHub repo and place necessary video files in ./input_data)
+ **[`utils.py`](utils.py)** - Utility functions used in scripts along the pipeline.

> **Timing** \
+ most time consuming step: mif_extraction_short
+ i5-7400 CPU @ 3.00GHz; 16GB RAM; No CUDA: >15h
+ Google Colab + GPU (CUDA) acceleration: 3-5h

## Folder-Structure

+ **./input_data**
Input Data (Video Files from MiT-v1, etc.)
+ **./models**
Folder containing labels and weights of the pretrained ResNet-50 (MiT-v1) neural network model, as well as representations of this/these model(s), e.g. RDM, feature maps, etc.
+ **./plots**
Output-folder, where plots, etc. are saved into during the process
+ **./scripts**
Main Py-Scripts and Py-Notebooks (:= core part of the pipeline)
+ **./temp**
Temporary data (like .csv, .pkg) for further use during running of the pipeline


## Steps

![Pipeline Flowchart](/plots/MIT_pipeline_1.1_wBG.png  "Processing Pipeline Steps")

### 1. Change Framerate
`00.change_framerate.py`

Prior to any further processing and analysis, conversion to final frame-rate has to be run.

### 2. MIF-Extraction
`01.mif_extraction_exploratory.ipynb`
`01.mif_extraction_short.ipynb`

Using ResNet50 (or any other model) pre-trained on Moments-in-Time version-1 (MiTv1) video dataset, the classification accuracies (outputs of last layer) are extracted per each frame of the pre-selected video set.

The script first sweeps through videos of the given video-set and extracts the softmax (classification accuracy) values for each frame using ResNet50 pre-trained on MiTv1. For example, a video of 75 frames, it will extract 75 x 339 values, where 339 is the number of labels/categories on which the ResNet50 was pre-trained on. These values are collected into a nested python dictionary structured in the following
way:
  ```
  {
    category_i :
    {
      file_name_j :
        list of length = n_frames :
          list of length = n_labels
    }
  }  
  ```
  This script contains additional features serving as exploratory steps.
  
  *Used* for further Hierarchical Clustering Analysis (HCA).

The script then sweeps again the same way, but instead of extracting softmax values for all labels, it only collects the those corresponding to the true category of the given video. Having these values for all the frames of a video, it finds the biggest one and extracts the index corresponding to it. This index corresponds to the **Most Informative Frame (MIF)** and is further collected into a pandas `DataFrame`:

  | category | fname | mif_idx | softmax[category] |
  | -------- |:-----:| :------:| ----------------: | 
  | aiming | yt-0gwUV4Ze-Hs_390.mp4	| 50 | 0.6277048587799072 |
  | ... | ... | ... | ...|
  |yawning | yt-_0YhIgRtYfo_3.mp4 | 23 | 0.9770082235336304 |
  
  *Used* for all further steps, except for HCA.

> **dictionary vs DataFrame** \
The DataFrame is a very small subset of values that can be found in the 
softmax dictionary, but incorporates all the information, i.e. MIF indexes,
needed for all further processing steps, except for HCA. On the other hand,
having the softmax dictionary, the MIF indexes can be easily extracted
(see `softmax_to_mif-idx`).  

### 3. RDM Creation
`02.softmax_rdm.py`

**HCA** for category reduction using softmax features vectors from RN50. Having the softmax dictionary from `RN50_classification` step, feature vectors are extracted for each category present in the given video set where each vector has the length equal to `n_labels` (=339) the model is pre-trained on. These feature vectors are used to compute a distance matrix that is further used to perform a hierarchical clustering.

Extracts feature vectors out of softmax dictionary and computes a distance matrix.
    
*For example*, the first iteration of the video set had 306 categories each containing a varying number of videos. The feature vectors were obtained by averaging the softmax vectors over the videos, resulting in a single vector per category. Iterating over all categories results in a 2D feature array of size `(306, 339)`. This array is further used to compute a distance matrix, using a given metric as, for example, `cosine distance`. As a result, a square matrix of shape `(306, 306)` is obtained which can be further used for additional analysis like HCA, or even Representational Similarity Analysis (RSA).

### 4. Hierarchical Cluster Analysis (HCA)
`03.hca.py`

Performs HCA using previously obtained softmax distance matrix.

Steps:
* Compute cophenetic correlation distance to find out the best linkage method;
* Compute Silhouette's Index to find the "best" number of clusters;
* Fit HC using best `n_clusters`;
* Plot dendrogram.

### 5. Softmax to MIF-Index
`03.softmax_to_mif-idx.py`

Extraction of MIF indexes, based on RN50 softmax accuracy dictionary. Indices stored in pandas data frame.

### 6. Resiszing & Cropping
`04.resize&crop.py`

Resizing and cropping of the video data via ffmpeg. Also includes checking the width x height (resolution) ratio via decord.

### 7. ResNet-50 Classification & Visualization
`05.classification_visualization.py`

Script for investigation of per-frame TopN accuracies extracted using an MiTv1 pretrained ResNet50.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CogNeuroUR/frame-by-frame/blob/main/video_frame_extractor.ipynb)

![example image](https://github.com/CogNeuroUR/frame-by-frame/blob/main/plots/best&worst+top5/cutting_yt-UgO4jE-puiE_67.mp4.png?raw=true)


* Having a collection of videos with pre-defined "true" categories and a pretrained classifier:
    1. extract each frame of each video;
    2. perform a classification of each individual frame;
    3. extract the **per-frame accuracies** for the true category.
* Frame extractor variants:
    * [decord](https://github.com/dmlc/decord) [implemented]:
        ```
        vr = decord.VideoReader()
        frame_id_list = range(start, stop, pace)
        video_frames = vr.get_batch(frame_id_list).asnumpy()
        ```
    * [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html) [TODO]

### 8. Create GIFS
`06.write_gifs.py`

Script for writting GIFs based on MIF indices.

Three different GIFs were extracted depending on the positioning to the MIF (most informative frame):
* '..._b.GIF' - MIF at the beginning, followed by 30 frames (Δt = 1s; FPS = 30 FPS (compare above))
* '..._m.GIF' - MIF in the middle of the GIF (+/- 15 frames)
* '..._e.GIF' - MIF at the end of the GIF (30 frames ending with GIF)

**Note:** Depending on the input video, sometimes not all the three different GIFs exist, due to the MIF index (and its position in the video).
(Comment: This was ok for us, as afterwards, there followed a manual selection process by human rators anyhow)

### 9. Rename Files
`07.renaming.py`

Script for renaming MIF, i.e. PNG, and MP4 datasets using the input GIF dataset.
The script was run for renaming (1) the original MP4s (, (2) the extracted MIF PNGs (see also below.)) and (3) the extracted GIFs (see above)

To have an overall consitent naming scheme, the videos were named as examplars form the category they were from.
E.g.: 'aiming_1.GIF', 'cutting_2.MP4' (or 'writing_3.PNG')

The conversion from original raw file names to renamed file names follwoing the pattern described above is additionaly stored and archived in a lookup-table.

### 10. Convert GIF to MP4
`08.convert_gif_to_mp4.py`

Script writting GIFs based on MIFs.

**Info:** [SoSciSurvey](https://www.soscisurvey.de/) (2021-04-20), has an upload quota for GIFs which is different than for MP4s (640 kB vs. 64 MB).

As using that platform for later batch human stimuli evaluation, re-transforming of the GIFs to videos was exectuted.
Note that these videos differ from the original videos (e.g. duration: 1s; pre-selection of MIF positioning in video; etc.)

### 11. Extract Static Frames
`09.extract_static_imgs.py`

Script extracting static images, i.e MIFs.
For extraction the module PIL and VideoReader are used.
This script may produce both JPGs oder PNGs. 

**Note:** Renaming might also be run again after this steps. The scripts might be also run in slightly differed order than displayed above.

### 12. Resolution Statistics
`10.collect_resolutions.py`

Collects the resolution (width) of videos in the video set. This script uses decord and cv2 for accomplishing the collection of resolution of input videos.
The final distributions are exported as csv-file for further visualization a./o. statistics.

### 13. GIF Selection Statistics
`11.gif_selection_behaviour.py`
`11.gif_selection_rators.py`

Extracts distribution of GIFs per MIF position for each participant in the GIF selection task (using Rator distribution specified).

In a step above, up to (!) three different GIFs were generated using the MIF indices. Four different human rators (2021-04-28) selected one of these up to three different possible GIFs.
To check individual rating behavior (for bias, overall tendencies over rators, etc.), the individual selections are collected, quantified and visualized.

The first script extracts the distribution of GIFs per MIF position for each participant in the GIF selection task sepcified (= rating).
The second script reads in a input csv (containing individual rating behavior / categories).
Both parts plot the distribution for manual review. Note that the second is closer to the actual behavior due to the more precise repproduction of actual human rating behavior (thrrough the input csv).















<!-- Frame-by-frame classification
=======================================================

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CogNeuroUR/frame-by-frame/blob/main/video_frame_extractor.ipynb)

![example image](https://github.com/CogNeuroUR/frame-by-frame/blob/main/plots/best&worst+top5/cutting_yt-UgO4jE-puiE_67.mp4.png?raw=true)


* Having a collection of videos with pre-defined "true" categories and a pretrained classifier:
    1. extract each frame of each video;
    2. perform a classification of each individual frame;
    3. extract the **per-frame accuracies** for the true category.
* Frame extractor variants:
    * [decord](https://github.com/dmlc/decord) [implemented]:
        ```
        vr = decord.VideoReader()
        frame_id_list = range(start, stop, pace)
        video_frames = vr.get_batch(frame_id_list).asnumpy()
        ```
    * [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html) [TODO]
  -->
