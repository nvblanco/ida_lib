.. toctree::
   :maxdepth: 2

Neural Net example
======================

.. code-block:: Python

    from __future__ import print_function

    import os
    import os.path
    import sys
    import torch
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn as nn
    import torch.nn.functional as F

    from ida_lib.core.pipeline_geometric_ops import HflipPipeline, RandomShearPipeline, \
        RandomRotatePipeline
    from ida_lib.core.pipeline_pixel_ops import NormalizePipeline, RandomContrastPipeline
    from ida_lib.image_augmentation.data_loader import AugmentDataLoader

    import kornia
    if sys.version_info[0] == 2:
        import cPickle as pickle
    else:
        import pickle

    import torch.utils.data as data
    from torchvision.datasets.utils import download_url, check_integrity

    '''https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network'''

    # create a custom cifar Dataset to read the data
    class custom_CIFAR10(data.Dataset):
        """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

        Args:
            root (string): Root directory of dataset where directory
                ``cifar-10-batches-py`` exists.
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.

        """
        base_folder = 'cifar-10-batches-py'
        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
        train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
            ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
            ['data_batch_4', '634d18415352ddfa80567beed471001a'],
            ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ]

        test_list = [
            ['test_batch', '40351d587109b95175f43aff81a1287e'],
        ]

        def __init__(self, root, train=True,
                     transform=None, target_transform=None,
                     download=False):
            self.root = os.path.expanduser(root)
            self.transform = transform
            self.target_transform = target_transform
            self.train = train  # training set or test set

            if download:
                self.download()

            if not self._check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You can use download=True to download it')

            # now load the picked numpy arrays
            if self.train:
                self.train_data = []
                self.train_labels = []
                for fentry in self.train_list:
                    f = fentry[0]
                    file = os.path.join(root, self.base_folder, f)
                    fo = open(file, 'rb')
                    if sys.version_info[0] == 2:
                        entry = pickle.load(fo)
                    else:
                        entry = pickle.load(fo, encoding='latin1')
                    self.train_data.append(entry['data'])
                    if 'labels' in entry:
                        self.train_labels += entry['labels']
                    else:
                        self.train_labels += entry['fine_labels']
                    fo.close()

                self.train_data = np.concatenate(self.train_data)
                self.train_data = self.train_data.reshape((50000, 3, 32, 32))
                self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            else:
                f = self.test_list[0][0]
                file = os.path.join(root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.test_data = entry['data']
                if 'labels' in entry:
                    self.test_labels = entry['labels']
                else:
                    self.test_labels = entry['fine_labels']
                fo.close()
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is index of the target class.
            """
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

            item = {'image': img, 'target': target}
            return item  # modified to return a dict instead of a tuple

        def __len__(self):
            if self.train:
                return len(self.train_data)
            else:
                return len(self.test_data)

        def _check_integrity(self):
            root = self.root
            for fentry in (self.train_list + self.test_list):
                filename, md5 = fentry[0], fentry[1]
                fpath = os.path.join(root, self.base_folder, filename)
                if not check_integrity(fpath, md5):
                    return False
            return True

        def download(self):
            import tarfile

            if self._check_integrity():
                print('Files already downloaded and verified')
                return

            root = self.root
            download_url(self.url, root, self.filename, self.tgz_md5)

            # extract file
            cwd = os.getcwd()
            tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
            os.chdir(root)
            tar.extractall()
            tar.close()
            os.chdir(cwd)

    #auxiliar function to plot batches images
    def plot_tuple_batch(images, labels):
        batch_size = images.shape[0]
        images = images.cpu()
        labels = labels.cpu()

        fig, axs = plt.subplots(1, batch_size, figsize=(16, 10))
        for i in range(batch_size):
            axs[i].axis('off')
            axs[i].set_title(classes[labels[i].item()])
            img: np.ndarray = kornia.tensor_to_image((images[i] * 255).byte())
            axs[i].imshow(img)
        plt.show()

    # initialize train dataset
    trainset = custom_CIFAR10(root='./data', train=True,
                              download=True)
    # define the cnn model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # def train loop
    def train():
        net = Net()
        net = net.cuda()

        # Configure parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # TRAIN
        from time import time
        start_time = time()
        for epoch in range(1):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = (inputs.float())
                labels = labels.to('cuda')
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        consumed_time = time() - start_time
        print(consumed_time)
        print('Finished Training')
        torch.save(net.state_dict(), PATH)

    #def test loop
    def test():
        images, labels = dataiter.next()

        # print images
        plot_tuple_batch(images, labels)
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
        net = Net()
        net = net.cuda()
        net.load_state_dict(torch.load(PATH))
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                      for j in range(4)))
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                labels = labels.to('cuda')
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

    # Create the dataloader with ida_lib augmentations
    trainloader = AugmentDataLoader(dataset=trainset,
                                    batch_size=4,
                                    shuffle=True,
                                    resize=(500, 500),
                                    pipeline_operations=(NormalizePipeline(probability=1),
                                                         HflipPipeline(probability=1),
                                                         RandomRotatePipeline(probability=0, degrees_range=(-15, 15)),
                                                         RandomContrastPipeline(probability=0, contrast_range=(0.8, 1.2)),
                                                         RandomShearPipeline(probability=0, shear_range=(0, 0.5))),
                                    interpolation='bilinear',
                                    padding_mode='zeros',
                                    output_format='tuple',
                                    output_type=torch.float32
                                    )
    # initialize test dataset
    testset = custom_CIFAR10(root='./data', train=False,
                             download=True)

    # Create the dataloader with ida_lib augmentations
    testloader = AugmentDataLoader(dataset=testset,
                                   batch_size=4,
                                   shuffle=False,
                                   pipeline_operations=(NormalizePipeline(probability=1),
                                                        HflipPipeline(probability=0.5),
                                                        RandomRotatePipeline(probability=0.8, degrees_range=(-15, 15)),
                                                        RandomContrastPipeline(probability=0, contrast_range=(0.8, 1.2)),
                                                        RandomShearPipeline(probability=0, shear_range=(0, 0.5))),
                                   interpolation='bilinear',
                                   padding_mode='zeros',
                                   output_format='tuple',
                                   output_type=torch.float32
                                   )
    # clases
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # path to save weights
    PATH = './cifar_net2.pth'


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # plot some items of train
    plot_tuple_batch(images, labels)

    # train the net
    train()

    # test the results
    test()