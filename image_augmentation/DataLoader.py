from _ctypes import Union
from torch.utils.data import Dataset, DataLoader
from core.pipeline import pipeline
from core.pipeline_operations import *


class DataAugmentDataLoader(ABC, DataLoader):
    ''' The DataAugmentDataLoader class implements a Pytorch DataLoader but groups it into one class:
            * The Dataset object that takes care of reading the data (methods must be implemented by the user)
            * The iterative DataLoader object that will serve as an input system for a neural network.
            * A pipeline that applies data image Augmentation operations over the input data.

        To make use of this class, it is necessary to overwrite the methods corresponding to the dataset class (init_dataset,
        len_dataet, get_item_dataset) to make a personalized reading of your data.
        The arguments you pass to your init_dataset method will have to be passed when you initialize your custom AugmentDataloader'''

    @abstractmethod
    def init_dataset(self, *args, **kwargs):
        '''(Abstract method to be implemented)
        Dataset initialization, called only once. It is useful to open files and read data that are not too large in memory in this
        method and store all the necessary parameters'''
        pass

    @abstractmethod
    def len_dataset(self, *args, **kwargs) -> int:
        '''(Abstract method to be implemented)
        Returns the number of elements in the dataset'''
        pass

    @abstractmethod
    def get_item_dataset(self, *args, **kwargs) -> Union[dict, np.ndarray]:
        '''(Abstract method to be implemented)
           Return an item from the dataset. If it includes compound data, it must be a dict with elements like:
            'image', 'keypoints', 'label'...
            Read pipeline object for more detailed information'''
        pass

    def pipe_through(self, item: dict):
        '''Private method that passes a data element through the pipeline of input pipeline_operations '''
        return self.pipeline(item, visualize=False)

    class inner_Dataset(Dataset):
        def __init__(self, outer: DataLoader, *args, **kwargs):
            self.outer = outer
            outer.init_dataset(*args, **kwargs)

        def __len__(self):
            return self.outer.len_dataset()

        def __getitem__(self, idx: int):
            if self.outer.pipeline is not None:
                return self.outer.pipe_through(self.outer.get_item_dataset(idx))
            else:
                return self.outer.get_item_dataset(idx)

    def __init__(self,
                 batch_size,
                 shuffle=True,
                 pipeline_operations=None,
                 resize=None,
                 interpolation: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 *args,
                 **kwargs):
        self.dataset = self.inner_Dataset(outer=self, *args, **kwargs)
        if pipeline_operations is not None:
            self.pipeline = pipeline(resize=resize,
                                     interpolation=interpolation,
                                     padding_mode=padding_mode,
                                     pipeline_operations=pipeline_operations)
        else:
            self.pipeline = None
        DataLoader.__init__(self, dataset=self.dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)


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
