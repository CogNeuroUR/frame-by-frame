  #%% Define and load model
  from pathlib import Path
  import torch
  import torch.nn as nn
  import torchvision.models as models

  # Define
  resnet50 = models.resnet50(pretrained=False, progress=True, num_classes=339)
  # Load pretrained weights (MiTv1)
  path_model = Path('model_zoo/resnet50_moments-fd0c4436.pth')
  resnet50.load_state_dict(torch.load(path_model))
  # Evaluation mode
  resnet50.eval()

  #%% Transformations
  import torchvision.transforms as transforms
  transformation = transforms.Compose([
                                      transforms.ToPILImage(mode='RGB'), # required if the input image is a nd.array
                                      transforms.Resize(224), # To be changed to rescale to keep the aspect ration?
                                      transforms.CenterCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
  ])

  # %% Load categories
  def load_categories():
      """Load categories."""
      with open(Path('labels/category_momentsv1.txt')) as f:
          return [line.rstrip() for line in f.readlines()]

  # load categories
  categories = load_categories()

  #%% Sweep through files in subfolders of path_input
  path_input = Path('data/MIT_sampleVideos_RAW_WORK_IN_PROGRESS').absolute()

  l_videos = []
  for path, subdirs, files in os.walk(path_input):
    for name in files:
      if name[-3:] == 'mp4':
        l_videos.append([path.split('/')[-1],   # category
                        name])                 # file name
      else:
        print('Ignored: ', name)

  if l_videos:
    l_videos = sorted(l_videos)
  print('Total nr. of MP4s: ', len(l_videos))

  # %% Sweep through videos
  import time
  import decord
  decord.bridge.set_bridge('native') # Seems to be the fastest option
  from decord import cpu
  from decord import VideoReader
  from torch.nn import functional as F
  import numpy as np

  vervbose = True

  start = time.time()

  j = 0

  for category, file_name in l_videos[:1]:
    # Verbose
    print(f'{j}/{len(l_videos)}'); j+=1

    cat_idx = categories.index(category)
    path_input_file = str(path_input / category/ file_name)
    
    # Load video with Decord.VideoReader
    vr = VideoReader(path_input_file)
    video_frames = vr.get_batch(range(0, len(vr), 1))
    
    pred_accuracies = np.zeros((video_frames.shape[0], ))
    
    for i in range(video_frames.shape[0]):
      input = transformation(video_frames.asnumpy()[i])
      
      # Classification:
      logit = resnet50.forward(input.unsqueeze(0))
      #h_x = F.softmax(logit, 1).data.squeeze().tolist()
      h_x = F.softmax(logit, 1).data.squeeze().numpy()[cat_idx]
      pred_accuracies[i]= h_x
    
    pred_accuracies = pred_accuracies
    print(np.argmax(pred_accuracies), pred_accuracies[np.argmax(pred_accuracies)])
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
    """
      
  stop = time.time()
  duration = stop-start
  print(f'\nTime spent: {duration:.2f}s (~ {duration/j:.2f}s per file)')

  # %%
  from pathlib import Path
  import pickle
  import numpy as np
  path_prefix = Path().parent.absolute()
  dict_path = path_prefix / 'saved/full/accuracies_per_category_full_mitv1.pkl'
  # Load from file
  f = open(dict_path, 'rb')
  accuracies_per_category = pickle.load(f)

  #%%
  l_categories = categories

  category_name = 'adult+female+singing'
  video_fname = 'yt-0rvJOREmAv4_60.mp4'

  per_frame_accuracies = np.array(accuracies_per_category[category_name][video_fname])
  ass = per_frame_accuracies.reshape((per_frame_accuracies.shape[0],
                                              len(list(accuracies_per_category.keys()))))

  cat_idx = [i for i in range(len(l_categories))
            if l_categories[i] == category_name][0]
  ass = ass[:, cat_idx]

  print(f'\t{video_fname} : Max/Min accuracy at frame:' \
  f' {np.argmax(ass)}/{np.argmin(ass)}' \
  f' with value: {ass[np.argmax(ass)]}' \
  f' / {ass[np.argmin(ass)]}')
  # %%
