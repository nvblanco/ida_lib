import numpy as np
import torch
from operations import geometry, color
import kornia
from matplotlib import image
import matplotlib.pyplot as plt
import cv2


# read the image with OpenCV
img: np.ndarray = cv2.imread('./gato.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

keypoints = ([img.shape[0]//2, img.shape[1]//2], [img.shape[0]//2  + 105, img.shape[1]//2 - 50])

# convert to torch tensor



data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)  # BxCxHxW
points = [torch.from_numpy(np.asarray(point)) for point in keypoints]

#data = color.change_brightness(img, 1.5)


from time import time



data = {'image':data, 'keypoints': points}

from operations import utils

utils.keypoints_to_homogeneus_and_concatenate(points)

#data_warped: torch.tensor = kornia.warp_affine(data.float(), M, dsize=(h, w))
#data_warped: torch.tensor = kornia.hflip(data_warped)
data_warped: torch.tensor = geometry.scale(data, 0.8, True)


center = torch.ones(1, 2)
center[..., 0] = img.shape[2] / 2  # x
center[..., 1] = img.shape[1] / 2  # y
alpha = 45.0
angle = torch.ones(1) * alpha
scale= torch.ones(1)
tr = kornia.get_rotation_matrix2d(center, angle, scale).to('cuda')
data_warped = geometry.vflip(data, True)
#data_warped = geometry.rotate(data, degrees=25,visualize=True)
#data_warped: torch.tensor = geometry.rotate(data_warped, 30)
# convert back to numpy
img_warped: np.ndarray = kornia.tensor_to_image(data_warped['image'].byte()[0])

# create the plot
fig, axs = plt.subplots(1, 2, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('image source')
xvalues = [value[0] for value in keypoints]
yvalues = [value[1] for value in keypoints]
axs[0].scatter(x = xvalues, y = yvalues, s=80)
axs[0].imshow(img)

axs[1].axis('off')
axs[1].set_title('image warped')
xvalues_warped = [value[0].numpy() for value in data_warped['keypoints']]
yvalues_warped = [value[1].numpy() for value in data_warped['keypoints']]
axs[1].scatter(x = xvalues_warped, y = yvalues_warped, s=80)
axs[1].imshow(img_warped)
plt.show()



'''
image = torch.rand(2, 50 , 50)
keypoints = (torch.rand(1,2), torch.rand(1,2))


mask = torch.rand(1,50,50)
for i in range(mask.shape[1]):
    for j in range(mask.shape[2]):
        if mask[0,i,j] < 0.3:
            mask[0,i,j] = 0
        else:
            mask[0,i,j] = 1
center = torch.ones(1, 2)
center[..., 0] = 25  # x
center[..., 1] = 25  # y
alpha = 4.0
angle = torch.ones(1) * alpha
scale= torch.ones(1)
tr = kornia.get_rotation_matrix2d(center, angle, scale)

point = torch.rand(1,2)
point = torch.cat((point, torch.ones(1,1)), axis = 1)



data = {"image": image, "mask": image, "keypoints": keypoints}
#image = geometry.preprocess_data(data)


center = torch.ones(1, 2)
center[..., 0] = 25  # x
center[..., 1] = 25 # y
alpha = 45.0
angle = torch.ones(1) * alpha
scale= torch.ones(1)
tr = kornia.get_rotation_matrix2d(center, angle, scale)
degrees = 20
data= kornia.geometry.rotate(image, angle=degrees*torch.ones(1))


'''