from skimage import io
from ida_lib.operations.utils import save_im
from tqdm import trange
import os
import csv
import pandas as pd
from torch.utils.data import Dataset
from ida_lib.core.pipeline import *


class AgmentToDisk(object):
    '''
    The AgmentToDisk object allows to perform Data Image Augmentation directly to disk. That is, to save the images generated to disk to be used in future processes.
    '''

    def __init__(self,
                 dataset:           Dataset,
                 samples_per_item:  int = 2,
                 operations:        Union[list, None] = None,
                 interpolation:     str = 'bilinear',
                 padding_mode:      str = 'zeros',
                 resize:            Union[tuple, None] = None,
                 output_extension:  str = '.jpg',
                 output_csv_path:   str = 'anotations.csv',
                 output_path:       str = './augmented'
                 ):
        '''

        :param dataset          (Dataset)   : input dataset in charge of reading the input data
        :param samples_per_item (int)       : number of desired output samples per input element
        :param output_extension (str)       : desired image extension for the generated images
            ( '.jpg' | '.png' | '.gif' | '.jpeg' ... )
        :param output_csv_path  (str)       : path to the csv file (if is needed) to save anotations of the augmented data
        :param output_path      (str)       : path to the directory in which to save the generated data
        :param operations       (list)      : list of pipeline initialized operations (see pipeline_operations.py)
        :param resize           (tuple)     : tuple of desired output size. Example (25,25)
        :param interpolation    (str)       : interpolation mode to calculate output values
            ('bilinear' | 'nearest') .              Default: 'bilinear'.
        :param padding_mode     (str)       : padding mode for outside grid values
            ('zeros' | 'border' | 'reflection'.)    Default: 'zeros'
        '''
        self.dataset = dataset
        self.samples_per_item = samples_per_item
        self.output_extension = output_extension
        self.output_path = output_path
        self.output_csv_path = output_csv_path
        self.output_csv = []
        self.types2d = None
        self.other_types = None
        self.pipeline = pipeline(interpolation=interpolation,
                                 padding_mode=padding_mode,
                                 resize=resize,
                                 pipeline_operations=operations)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def save_item(self, item: dict, index: int, output_path: str, types_2d: list, other_types: list):
        '''
            ***This method can be overwritten to make a customized saving of the items according to the interests of the user***
        Method that implements the way to save to disk each of the generated elements. By default it saves all the generated
        images in the specified path. The samples are organized by name following the form:
            * images:                       <id_image>_<sample number> <extension>
            * other two-dimensional types:  <id_image>-<data_type>_<sample number> <extension>
        Annotations on the data, such as labels, or point coordinates are stored in dictionaries that will be written when all
        the images have been processed.

        :param item         (dict)  : input element to be saved to disk
        :param index        (int)   : sample number to which the input item corresponds
        :param output_path  (str)   : path to the directory in which to save the generated data
        :param types_2d     (list)  : list of types of two dimensional data of the input item
        :param other_types  (list)  : list of types that are not two-dimensional elements
        '''
        item['id'] = item['id'] + '_' + str(index)
        for type in types_2d:
            if 'image' in type:
                name = output_path + '/' + item['id']
            else:
                name = output_path + item['id'] + '-' + type + '_' + str(index)
            save_im(tensor=item[type], title=(name + self.output_extension))
        self.output_csv.append(dict((label, item[label]) for label in (other_types)))

    def final_save(self):
        '''
            ***This method can be overwritten to make a customized saving of the items according to the interests of the user***
        Method that runs only once, once all the images have been processed. Useful for writing csv with image annotations. By default
        the annotations of all images are saved in the same file. The csv file will have one row for each generated element, identified
        by its id. Each column will correspond with the labels associated to each generated element. In the case of coordinate lists,
        their coordinates are arranged in columns separating the x and y coordinates in each element (point0_x, point0_y, point1_x, ..., pointn_y)
        '''
        csv_columns = self.output_csv[0].keys()
        try:
            with open(self.output_csv_path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in self.output_csv:
                    writer.writerow(data)
        except IOError:
            print("I/O error")

    def __call__(self, *args, **kwargs):
        pbar = trange(len(self.dataset))
        for i in pbar:
            pbar.set_description("Processing image " + str(i) + " of the input dataset ")
            item = [self.dataset[i] for _ in range(self.samples_per_item)]
            augmented = self.pipeline(item)
            self.types2d, self.other_types = self.pipeline.get_data_types()
            if self.types2d is None:
                self.types2d, self.other_types = self.pipeline.get_data_types()
            for index, item in enumerate(augmented):
                self.save_item(item, index, types_2d=self.types2d, other_types=self.other_types,
                               output_path=self.output_path)
        self.final_save()
        total_images = len(self.dataset) * self.samples_per_item
        print("Generated " + str(total_images) + " new images from the original dataset.")


# samples_per_item = 5


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
        id = (self.landmarks_frame.iloc[idx, 0]).split('.')[0]
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'id': id, 'image': image, 'landmarks': landmarks}

        return sample


face_dataset = FaceLandmarksDataset(csv_file='../../examples/faces/face_landmarks.csv',
                                    root_dir='../../examples/faces/')

augmentor = AgmentToDisk(dataset=face_dataset,
                         samples_per_item=50,
                         operations=(RandomScalePipeline(probability=0.6, scale_range=(0.8, 1.2), center_desviation=20),
                                     HflipPipeline(probability=0.5),
                                     RandomContrastPipeline(probability=0.5, contrast_range=(1, 1.5))),
                         interpolation='nearest',
                         padding_mode='zeros',
                         resize=None,
                         output_extension='.jpg',
                         output_csv_path='anotations.csv',
                         output_path='./augmented_custom')

augmentor()

'''output_csv = []
output_path = './augmented'
extension = '.jpg'


def save_item(item, index, output_path, types_2d, other_types):
    item['id'] = item['id'] + '_' + str(index)
    labels_dict = {}
    for type in types_2d:
        if 'image' in type:
            name = output_path + '/' + item['id']
        else:
            name = output_path + item['id'] + '-' + type + '_' + str(index)
        save_im(tensor=item[type], title=(name + extension))
    for label in other_types:
        labels_dict.update(element_to_dict_csv_format(item[label], label))
    output_csv.append(labels_dict)


def final_save(output_csv_path='anotations.csv'):
    csv_columns = output_csv[0].keys()
    try:
        with open(output_csv_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in output_csv:
                writer.writerow(data)
    except IOError:
        print("I/O error")

samples_per_item = 5
face_dataset = FaceLandmarksDataset(csv_file='../faces/face_landmarks.csv',
                                    root_dir='../faces/')

pip = pipeline(interpolation='nearest', pipeline_operations=(
    translatePipeline(probability=0.5, translation=(3, 1)),
    vflipPipeline(probability=0.5),
    hflipPipeline(probability=0.5),
    contrastPipeline(probability=0.5, contrast_factor=1),
    randomBrightnessPipeline(probability=0.5, brightness_range=(1, 1.2)))
               )

if not os.path.exists(output_path):
    os.makedirs(output_path)
pbar = trange(len(face_dataset))
for i in pbar:
    pbar.set_description("Processing image " + str(i) + " of the input dataset ")
    item = [face_dataset[i] for _ in range(samples_per_item)]
    augmented = pip(item)
    types2d, other_types = pip.get_data_types()
    for index, item in enumerate(augmented):
        save_item(item, index, types_2d=types2d, other_types=other_types, output_path=output_path)
final_save()
'''
