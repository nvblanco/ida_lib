import test
import numpy as np
import cv2
import torch
import kornia

img: np.ndarray = cv2.imread('../gato.jpg')

img2: np.ndarray = cv2.imread('../bird.jpg')

# read the image with OpenCV
img: np.ndarray = cv2.imread('../gato.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

keypoints = ([img.shape[0]//2, img.shape[1]//2], [img.shape[0]//2  + 105, img.shape[1]//2 - 50])

# convert to torch tensor
data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)  # BxCxHxW
points = [torch.from_numpy(np.asarray(point)) for point in keypoints]

data = {'image':data, 'keypoints': points}


#test.visualize([data, data], [data, data])

test.visualize_test()