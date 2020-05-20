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
    translate_pipeline(probability=0, translation=(3, 1)),
    vflip_pipeline(probability=1),
    hflip_pipeline(probability=1),
    contrast_pipeline(probability=0, contrast_factor=1),
    random_brightness_pipeline(probability=0, brightness_range=(1, 1.2)),
    gamma_pipeline(probability=0, gamma_factor=0),
    random_translate_pipeline(probability=0, translation_range=(-90, 90)),
    random_scale_pipeline(probability=0, scale_range=(0.5, 1.5), center_desviation=20),
    random_rotate_pipeline(probability=0, degrees_range=(-50, 50), center_desviation=20),
    random_translate_pipeline(probability=0, translation_range=(20, 100)),
    random_shear_pipeline(probability=0, shear_range=(0, 0.5))
))

dataloader = test_dataloader(batch_size=4,
                             num_workers=1,
                             shuffle=True,
                             pipeline_operations=(
                                 translate_pipeline(probability=0, translation=(3, 1)),
                                 vflip_pipeline(probability=1),
                                 hflip_pipeline(probability=1)),
                             resize=None,
                             interpolation='bilinear',
                             padding_mode='zeros',
                             csv_file='faces/face_landmarks.csv',
                             root_dir='faces/'
                             )

# sample = face_dataset[1]
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())
print('holi')
