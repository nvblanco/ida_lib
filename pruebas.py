from image_augmentation.DataLoader import *

'''class test_dataset(custom_Dataset):
    def __init__(self, pipeline, csv_file, root_dir):
        custom_Dataset.__init__(self, pipeline)
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def custom_getitem(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': landmarks}
        return sample'''


class test_dataloader(augment_dataLoader):
    def init_dataset(self, csv_file, root_dir):
        #custom_Dataset.__init__(self)
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def len_dataset(self):
        return len(self.landmarks_frame)

    def get_item_dataset(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': landmarks}
        return sample


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


pip = pipeline(interpolation='nearest', pipeline_operations=(
    translatePipeline(probability=0, translation=(3, 1)),
    vflipPipeline(probability=1),
    hflipPipeline(probability=1),
    contrastPipeline(probability=0, contrast_factor=1),
    randomBrightnessPipeline(probability=0, brightness_range=(1, 1.2)),
    gammaPipeline(probability=0, gamma_factor=0),
    randomTranslatePipeline(probability=0, translation_range=(-90, 90)),
    randomScalePipeline(probability=0, scale_range=(0.5, 1.5), center_desviation=20),
    randomRotatePipeline(probability=0, degrees_range=(-50, 50), center_desviation=20),
    randomTranslatePipeline(probability=0, translation_range=(20, 100)),
    randomShearPipeline(probability=0, shear_range=(0, 0.5))
))


#torch.multiprocessing.set_start_method('spawn')
dataloader = test_dataloader(batch_size=4,
                             num_workers=0,
                             shuffle=True,
                             pipeline_operations=(
                                 translatePipeline(probability=0, translation=(3, 1)),
                                 vflipPipeline(probability=1),
                                 hflipPipeline(probability=1)),
                             resize=(500, 326),
                             interpolation='bilinear',
                             padding_mode='zeros',
                             csv_file='faces/face_landmarks.csv',
                             root_dir='faces/'
                             )

# sample = face_dataset[1]
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, )
print('holi')
