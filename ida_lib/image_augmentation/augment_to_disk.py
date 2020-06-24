import csv
import os

from torch.utils.data import Dataset
from tqdm import trange

from ida_lib.core.pipeline import *
from ida_lib.operations.utils import save_im, get_data_types, generate_dict


class AugmentToDisk(object):
    """
    The AugmentToDisk object allows to perform Data Image Augmentation directly to disk. That is, to save the images
    generated to disk to be used in future processes.
    """

    def __init__(self,
                 dataset: Dataset,
                 samples_per_item: Optional[int] = 2,
                 total_output_samples: Optional[int] = None,
                 operations: Union[list, None] = None,
                 interpolation: str = 'bilinear',
                 padding_mode: str = 'zeros',
                 resize: Union[tuple, None] = None,
                 output_extension: str = '.jpg',
                 output_csv_path: str = 'annotations.csv',
                 output_path: str = './augmented'
                 ):
        """

        :param dataset : input dataset in charge of reading the input data
        :param samples_per_item: number of desired output samples per input element
        :param total_output_samples: number of desired output total samples  (Optional)
        :param output_extension: desired image extension for the generated images
            ( '.jpg' | '.png' | '.gif' | '.jpeg' ... )
        :param output_csv_path : path to the csv file (if is needed) to save annotations of the augmented data
        :param output_path : path to the directory in which to save the generated data
        :param operations  : list of pipeline initialized operations (see pipeline_operations.py)
        :param resize : tuple of desired output size. Example (25,25)
        :param interpolation: interpolation mode to calculate output values
            ('bilinear' | 'nearest') .              Default: 'bilinear'.
        :param padding_mode : padding mode for outside grid values
            ('zeros' | 'border' | 'reflection'.)    Default: 'zeros'
        """
        self.dataset = dataset
        if samples_per_item < 1:
            raise TypeError("non-positive values are not allowed for 'samples_per_item'")
        self.samples_per_item = samples_per_item
        if total_output_samples:
            self.samples_per_item = total_output_samples // len(self.dataset)
        self.output_extension = output_extension
        self.output_path = output_path
        self.output_csv_path = output_csv_path
        self.output_csv = []
        self.types2d = None
        self.other_types = None
        self.pipeline = Pipeline(interpolation=interpolation,
                                 padding_mode=padding_mode,
                                 resize=resize,
                                 pipeline_operations=operations)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def save_item(self, item: dict, index: int, output_path: str, types_2d: list, other_types: list, element: int):
        """
          **This method can be overwritten to make a customized saving of the items according
          to the interests of the user**
        Method that implements the way to save to disk each of the generated elements. By default it saves all the
        generated images in the specified path. The samples are organized by name following the form:
            * images:                       <id_image>_<sample number> <extension>
            * other two-dimensional types:  <id_image>_<sample number>-<data_type><extension>
        Annotations on the data, such as labels, or point coordinates are stored in dictionaries that will be written
        when all  the images have been processed.

        :param item:        input element to be saved to disk
        :param element:      input element number to identify it
        :param index:       sample number to which the input item corresponds
        :param output_path: path to the directory in which to save the generated data
        :param types_2d:    list of types of two dimensional data of the input item
        :param other_types: list of types that are not two-dimensional elements
        """
        if 'id' not in item:
            item['id'] = 'item-' + str(element)
        for actual_type in types_2d:
            if 'image' in actual_type:
                name = output_path + '/' + item['id'] + '_' + str(index)
            else:
                name = output_path + '/' + item['id'] + '_' + str(index) + '-' + actual_type
            save_im(tensor=item[actual_type], title=(name + self.output_extension))
        item['id'] = item['id'] + '_' + str(index)
        item_csv = {}
        for label in other_types:
            item_csv.update(generate_dict(item, label))
        self.output_csv.append(item_csv)

    def final_save(self):
        """
          **This method can be overwritten to make a customized saving of the items according
          to the interests of the user**
        Method that runs only once, once all the images have been processed. Useful for writing csv with image
        annotations. By default the annotations of all images are saved in the same file. The csv file will have one
        row for each generated element, identified by its id. Each column will correspond with the labels associated
        to each generated element. In the case of coordinate lists, their coordinates are arranged in columns
        separating the x and y coordinates in each element (point0_x, point0_y, point1_x, ..., point_y)
        """
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
            self.types2d, self.other_types = get_data_types(item[0])
            if self.types2d is None:
                self.types2d, self.other_types = self.pipeline.get_data_types()
            if isinstance(augmented, dict):
                augmented = [augmented]
            for index, item in enumerate(augmented):
                self.save_item(item, index, types_2d=self.types2d, other_types=self.other_types,
                               output_path=self.output_path, element=i)
        self.final_save()
        total_images = len(self.dataset) * self.samples_per_item
        print("Generated " + str(total_images) + " new images from the original dataset.")
