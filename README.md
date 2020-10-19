Frame-by-frame classification
=======================================================

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CogNeuroUR/frame-by-frame/blob/main/video_frame_extractor.ipynb)

![example image](https://github.com/CogNeuroUR/frame-by-frame/blob/main/plots/best_vs_worst_frame_example.jpeg?raw=true)

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
 
