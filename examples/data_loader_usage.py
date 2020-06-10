'''
This file has an example of how to use IDALib's own DataLoader which includes a pipeline to perform image data
augmentation on your data.
This code follows the pytorch example of  of using a dataloader
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html  but adapted to the ida-lib dataloader
'''

import os
import kornia
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io

from ida_lib.core.pipeline_geometric_ops import TranslatePipeline, VflipPipeline, HflipPipeline, RandomShearPipeline
from ida_lib.core.pipeline_pixel_ops import ContrastPipeline
from ida_lib.image_augmentation.data_loader import *


#Firstly create custom dataset to read the input data
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'id': self.landmarks_frame.iloc[idx, 0], 'image': image, 'keypoints': landmarks}
        return sample

#Auxiliar function to display elements
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    img= kornia.tensor_to_image(image.byte())
    plt.imshow(img)
    landmarks = landmarks.cpu().numpy()
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='o', c='r')
    plt.show()
    plt.pause(0.001)  # pause a bit so that plots are updated


#initialize custom dataset
face_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                    root_dir='faces/')

#initialite the custom dataloader
dataloader = AugmentDataLoader(dataset=face_dataset,
                               batch_size=1,
                               shuffle=True,
                               pipeline_operations=(
                                        TranslatePipeline(probability=1, translation=(30, 10)),
                                        VflipPipeline(probability=0.5),
                                        HflipPipeline(probability=0.5),
                                        ContrastPipeline(probability=0.5, contrast_factor=1),
                                        RandomShearPipeline(probability=0.5, shear_range=(0, 0.5))),
                               resize=(500, 500), #we must indicate the size of the resize because the input images are not all the same size
                               interpolation='bilinear',
                               padding_mode='zeros'
                               )

number_of_iterations = 3 #number of times the entire dataset is processed
for epoch in range(number_of_iterations-1):
    for i_batch, sample_batched in enumerate(dataloader): #our dataloader works like a normal dataloader
        print(i_batch, )
        keypoints = sample_batched['keypoints'][0,:,:]
        show_landmarks(sample_batched['image'][0], keypoints)
    print('all elements of the original dataset have been displayed and processed')