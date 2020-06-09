from torch.utils.data import Dataset, DataLoader

from ida_lib.core.pipeline import pipeline


class DataAugmentDataLoader(DataLoader):
    """ The DataAugmentDataLoader class implements a Pytorch DataLoader but groups it into one class:
            * The Dataset object that takes care of reading the data (methods must be implemented by the user)
            * The iterative DataLoader object that will serve as an input system for a neural network.
            * A pipeline that applies data image Augmentation operations over the input data.

        To make use of this class, it is necessary to overwrite the methods corresponding to the dataset class (init_dataset,
        len_dataet, get_item_dataset) to make a personalized reading of your data.
        The arguments you pass to your init_dataset method will have to be passed when you initialize your custom AugmentDataloader"""

    class inner_Dataset(Dataset):
        def _pipe_through(self, item: dict):
            """Method that passes a data element through the pipeline of input pipeline_operations """
            return self.pipeline(item, visualize=False)

        def __init__(self, pipeline, dataset, *args, **kwargs):
            self.pipeline = pipeline
            self.input_dataset = dataset

        def __len__(self):
            return len(self.input_dataset)

        def __getitem__(self, idx: int):
            if pipeline is not None:
                return self._pipe_through(self.input_dataset.__getitem__(idx))
            else:
                return self.input_dataset._getitem_(idx)

    def __init__(self,
                 batch_size,
                 dataset: Dataset,
                 shuffle=True,
                 pipeline_operations=None,
                 resize=None,
                 interpolation: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 *args,
                 **kwargs):

        if pipeline_operations is not None:
            self.pipeline = pipeline(resize=resize,
                                     interpolation=interpolation,
                                     padding_mode=padding_mode,
                                     pipeline_operations=pipeline_operations)
        else:
            self.pipeline = None
        if dataset:
            self.dataset = self.inner_Dataset(pipeline=self.pipeline, dataset=dataset)
        DataLoader.__init__(self, dataset=self.dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)

