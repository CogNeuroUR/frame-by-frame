# MIT-Pipeline

## Prerequirements / Initial Info

+ utils.py - Comment on which functions in there, for what used (and when)?
+ Timing - Comment on What is the bottlenck out of these steps?
+ **platform independence**: tested under independently on CoLab, Mac and Linux = OK
+ **paths** are relative (clone GitHub repo and place necessary video files in ./input (?))


## Folder-Structure

+ ./input
+ ./models
+ ./plots
+ ./scripts
+ ./temp

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

Extraction of MIF indexes, based on RN50 softmax accuracy dictionary.

### 6. Resiszing & Cropping
`04.resize&crop.py`

Resizing and cropping of the video data.

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

### 9. Rename Files
`07.renaming.py`

Script for renaming MIF, i.e. PNG, and MP4 datasets using the input GIF dataset.

### 10. Convert GIF to MP4
`08.convert_gif_to_mp4.py`

Script writting GIFs based on MIFs.

### 11. Extract Static Frames
`09.extract_static_imgs.py`

Script extracting static images, i.e MIFs.

### 12. Resolution Statistics
`10.collect_resolutions.py`

Collects the resolution (width) of videos in the video set.

### 13. GIF Selection Statistics
`11.gif_selection_behaviour.py`
`11.gif_selection_rators.py`

Extracts distribution of GIFs per MIF position for each participant in the GIF selection task (using Rator distribution specified).









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
