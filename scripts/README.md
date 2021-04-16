## Chronological order of scripts

change 
* `change_framerate`

    
    Prior to any further processing and analysis, conversion to final frame-rate
    has to be run.

* `classification / mif extraction`
    
    Using ResNet50 (or any other model) pre-trained on Moments-in-Time version-1
    (MiTv1) video dataset, the classification accuracies (outputs of last layer) 
    are extracted per each frame of the pre-selected video set.
    Two scripts were written to fully or partially fulfill this task:
    * `RN50_classification` sweeps through videos of the given video-set and
    extracts the softmax (classification accuracy) values for each frame using 
    ResNet50 pre-trained on MiTv1.
    For example, a video of 75 frames, it will extract 75 x 339 values, where 339
    is the number of labels/categories on which the ResNet50 was pre-trained on.
    These values are collected into a nested python dictionary structured in the following
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


    * `mif_extraction` sweeps the same way `RN50_classification` does, but instead
    of extracting softmax values for all labels, it only collects the those corresponding
    to the true category of the given video.
    Having these values for all the frames of a video, it finds the biggest one
    and extracts the index corresponding to it.
    This index corresponds to the **Most Informative Frame (MIF)** and is further
    collected into a pandas `DataFrame`:

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

* **`hierarchical clustering analysis (HCA)`**
  
  **HCA** for category reduction using softmax features vectors from RN50.
  Having the softmax dictionary from `RN50_classification` step, feature vectors
  are extracted for each category present in the given video set where each vector 
  has the length equal to `n_labels` (=339) the model is pre-trained on.
  These feature vectors are used to compute a distance matrix that is further
  used to perform a hierarchical clustering.
  
  * `softmax_rdm`

    Extracts feature vectors out of softmax dictionary and computes a distance matrix.
    
    > *For example*, the first iteration of the video set had 306 categories,
    each containing a varying number of videos.
    The feature vectors were obtained by averaging the softmax vectors over the videos, 
    resulting in a single vector per category.
    Iterating over all categories results in a 2D feature array of size `(306, 339)`.
    This array is further used to compute a distance matrix, using a given metric
    as, for example, `cosine distance`.
    As a result, a square matrix of shape `(306, 306)` is obtained which can be further
    used for additional analysis like HCA, or even Representational Similarity Analysis (RSA).
  
  * `hca`
    
    Performs HCA using previously obtained softmax distance matrix.

    Steps:
    * Compute cophenetic correlation distance to find out the best linkage method;
    * Compute Silhouette's Index to find the "best" number of clusters;
    * Fit HC using best `n_clusters`;
    * Plot dendrogram.
  
  * `semantic_analysis` (**TODO**)

    Using RSA to compute correlation between category and semantic models' 
    representations.
    Example of semantic models:
    [BERT](https://huggingface.co/transformers/model_doc/bert.html) (mini and base), 
    [USE](https://tfhub.dev/google/universal-sentence-encoder/4), etc.

  * `tanglegrams` (**TODO**)
  
    Using category and semantic model RDMs, compute and plot tanglegrams to
    compare organization and overlap of category structure.
