import os
from skimage import io
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import pandas as pd
from torch.utils.data import Dataset, DataLoader

from core.pipeline import pipeline
from core.pipeline_operations import *


class augment_dataLoader(DataLoader):

    @abstractmethod
    def init_dataset(self,  *args, **kwargs):
        pass

    @abstractmethod
    def len_dataset(self,  *args, **kwargs):
        pass

    @abstractmethod
    def get_item_dataset(self,  *args, **kwargs):
        pass

    def pipe_through(self, item):
        return self.pipeline(item, visualize=False)

    class inner_Dataset(Dataset):
        def __init__(self, outer,  *args, **kwargs):
            self.outer = outer
            outer.init_dataset(*args, **kwargs)

        def __len__(self):
            return self.outer.len_dataset()

        def __getitem__(self,  idx):
            if self.outer.pipeline is not None:
                return self.outer.pipe_through(self.outer.get_item_dataset(idx))
            else:
                return self.outer.get_item_dataset(idx)

    def __init__(self,
                 batch_size,
                 num_workers,
                 shuffle=True,
                 pipeline_operations=None,
                 resize=None,
                 interpolation: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 *args,
                 **kwargs):
        self.dataset = self.inner_Dataset(outer = self, *args, **kwargs)
        if pipeline_operations is not None:
            self.pipeline = pipeline(resize=resize,
                                     interpolation=interpolation,
                                     padding_mode=padding_mode,
                                     pipeline_operations=pipeline_operations)
        else:
            self.pipeline = None
        sample = self.dataset[1]
        DataLoader.__init__(self, dataset=self.dataset,  batch_size=batch_size, num_workers= num_workers, shuffle= shuffle)







'''class custom_Dataset(Dataset):
     @abstractmethod
     def __init__(self, pipeline):
        self.pipeline = pipeline

     @abstractmethod
     def __len__(self):
        pass

     @abstractmethod
     def custom_getitem(self, idx):
         pass

     def __getitem__(self, idx):
        return self.pipeline(self.custom_getitem(idx), visualize=True)
'''



'''fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break'''