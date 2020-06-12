from ida_lib.image_augmentation.data_loader import *


class test_dataloader(AugmentDataLoader):
    def init_dataset(self, csv_file, root_dir):
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


pip = Pipeline(interpolation='nearest', pipeline_operations=(
    TranslatePipeline(probability=0, translation=(3, 1)),
    VflipPipeline(probability=1),
    HflipPipeline(probability=1),
    ContrastPipeline(probability=0, contrast_factor=1),
    RandomBrightnessPipeline(probability=0, brightness_range=(1, 1.2)),
    GammaPipeline(probability=0, gamma_factor=0),
    RandomTranslatePipeline(probability=0, translation_range=(-90, 90)),
    RandomScalePipeline(probability=0, scale_range=(0.5, 1.5), center_deviation=20),
    RandomRotatePipeline(probability=0, degrees_range=(-50, 50), center_deviation=20),
    RandomTranslatePipeline(probability=0, translation_range=(20, 100)),
    RandomShearPipeline(probability=0, shear_range=(0, 0.5))
))

dataloader = test_dataloader(batch_size=4,
                             shuffle=True,
                             pipeline_operations=(
                                 TranslatePipeline(probability=0, translation=(3, 1)),
                                 VflipPipeline(probability=1),
                                 HflipPipeline(probability=1)),
                             resize=(500, 326),
                             interpolation='bilinear',
                             padding_mode='zeros',
                             csv_file='examples/faces/face_landmarks.csv',
                             root_dir='examples/faces/'
                             )

# sample = face_dataset[1]
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, )
print('holi')
