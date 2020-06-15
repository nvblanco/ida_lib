.. toctree::
   :maxdepth: 2

Image Augmentation to Disk example
======================

.. code-block:: Python


    import os

    import numpy as np
    import pandas as pd
    import torch
    from skimage import io
    from torch.utils.data import Dataset

    from ida_lib.core.pipeline_geometric_ops import RandomScalePipeline, HflipPipeline
    from ida_lib.core.pipeline_pixel_ops import RandomContrastPipeline
    from ida_lib.image_augmentation.augment_to_disk import AugmentToDisk


    # Create custom dataset to read the input data to be augmented
    class FaceLandmarksDataset(Dataset):
        """Face Landmarks dataset."""

        def __init__(self, csv_file, root_dir, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """
            self.landmarks_frame = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.landmarks_frame)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_name = os.path.join(self.root_dir,
                                    self.landmarks_frame.iloc[idx, 0])
            item_id = (self.landmarks_frame.iloc[idx, 0]).split('.')[0]
            image = io.imread(img_name)
            landmarks = self.landmarks_frame.iloc[idx, 1:]
            landmarks = np.array([landmarks])
            landmarks = landmarks.astype('float').reshape(-1, 2)
            sample = {'id': item_id, 'image': image, 'landmarks': landmarks}
            return sample


    # Inicialize the custom datset

    face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                        root_dir='faces/')

    # parameter setting and initialization

    augmentor = AugmentToDisk(dataset=face_dataset,  # custom dataset that provides the input data
                              samples_per_item=5,  # number of samples per imput item
                              operations=(RandomScalePipeline(probability=0.6, scale_range=(0.8, 1.2), center_deviation=20),
                                         HflipPipeline(probability=0.5),
                                         RandomContrastPipeline(probability=0.5, contrast_range=(1, 1.5))),
                              interpolation='nearest',
                              padding_mode='zeros',
                              resize=(250, 250),  # Here resizing is necessary because the input images have different sizes
                              output_extension='.jpg',
                              output_csv_path='anotations.csv',
                              output_path='./augmented_custom')

    augmentor()  # Run the augmentation