from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from ida_lib.core.pipeline import Pipeline


class AugmentDataLoader(DataLoader):
    """ The DataAugmentDataLoader class implements a Pytorch DataLoader but groups it into one class:
            * The Dataset object that takes care of reading the data
            * The iterative DataLoader object that will serve as an input system for a neural network.
            * A pipeline that applies data image Augmentation operations over the input data.

        To make use of this class, it is necessary to provide a dataset to make a personalized reading of your data."""

    class InnerDataset(Dataset):
        """
        inner dataset is an internal class that uses the DataAugmentDataLoader to add the pipeline to the input dataset
        """

        def _pipe_through(self, item: dict):
            """Method that passes a data element through the pipeline of input pipeline_operations """
            return self.pipeline(item, visualize=False)

        def __init__(self, pipeline, dataset):
            self.pipeline = pipeline
            self.input_dataset = dataset

        def __len__(self):
            return len(self.input_dataset)

        def __getitem__(self, idx: int):
            if Pipeline is not None:
                return self._pipe_through(self.input_dataset.__getitem__(idx))
            else:
                # return self.input_dataset._getitem_(idx)
                return self.input_dataset[idx]

    def __init__(self,
                 batch_size,
                 dataset: Dataset,
                 shuffle=True,
                 pipeline_operations=None,
                 resize=None,
                 interpolation: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 output_format: str = 'dict',
                 output_type: Optional[torch.dtype] = None
                 ):

        # Initialize the internal pipeline
        if pipeline_operations is not None:
            self.pipeline = Pipeline(resize=resize,
                                     interpolation=interpolation,
                                     padding_mode=padding_mode,
                                     pipeline_operations=pipeline_operations,
                                     output_format=output_format,
                                     output_type=output_type)
        else:
            self.pipeline = None
        self.dataset = self.InnerDataset(pipeline=self.pipeline, dataset=dataset)
        DataLoader.__init__(self, dataset=self.dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)
