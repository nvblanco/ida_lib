from abc import ABC, abstractmethod

from torch.utils.data import Dataset, DataLoader

from core.pipeline import pipeline


class augment_dataLoader(ABC):

    @abstractmethod
    def init_dataset(self):
        pass

    @abstractmethod
    def len_dataset(self):
        pass

    @abstractmethod
    def get_item_dataset(self):
        pass

    def pipe_through(self, item):
        return self.pipeline(item, visualize=False)

    class inner_Dataset(Dataset):
        def __init__(self):
            augment_dataLoader.init_dataset(self)

        @abstractmethod
        def __len__(self):
            return augment_dataLoader.len_dataset(self)

        def __getitem__(self, idx):
            return augment_dataLoader.pipe_through(self,augment_dataLoader.get_item_dataset(self))

    def __init__(self,
                 batch_size,
                 num_workers,
                 shuffle=True,
                 pipeline_operations=None,
                 resize=None,
                 interpolation: str = 'bilinear',
                 padding_mode: str = 'zeros'):

        if pipeline_operations is not None:
            self.pipeline = pipeline(resize=resize,
                                     interpolation=interpolation,
                                     padding_mode=padding_mode,
                                     pipeline_operations=pipeline_operations)

        self.dataset = self.inner_Dataset()
        self.dataLoader = DataLoader(self.dataset, batch_size=batch_size,
                   shuffle=shuffle, num_workers=num_workers)



