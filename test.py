import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch

from ida_lib.operations import transforms

# read the image with OpenCV
img: np.ndarray = cv2.imread('./gato.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

keypoints = ([img.shape[0] // 2, img.shape[1] // 2], [img.shape[0] // 2 + 15, img.shape[1] // 2 - 50],
             [img.shape[0] // 2 + 85, img.shape[1] // 2 - 80], [img.shape[0] // 2 - 105, img.shape[1] // 2 + 60])

# convert to torch tensor


points = [torch.from_numpy(np.asarray(point)) for point in keypoints]
data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)  # BxCxHxW
keypoints = ([img.shape[0] // 2, img.shape[1] // 2], [img.shape[0] // 2 + 15, img.shape[1] // 2 - 50],
             [img.shape[0] // 2 + 85, img.shape[1] // 2 - 80], [img.shape[0] // 2 - 105, img.shape[1] // 2 + 60])
points = [torch.from_numpy(np.asarray(point)) for point in keypoints]

# data = color.equalize_histogram(data, visualize=True)

data = {'image': data, 'keypoints': points}
matrix = torch.eye(2, 3).to('cuda')

center = torch.ones(1, 2)
center[..., 0] = data['image'].shape[-2] // 2  # x
center[..., 1] = data['image'].shape[-1] // 2  # y

# data = geometry.translate(data, visualize = False, translation = (20,-10))
# data = geometry.scale(data, visualize = False, scale_factor=0.75)
data = transforms.change_contrast(data, contrast=1.2, visualize=True)
# data = geometry.affine(data, visualize=False, matrix=matrix)
# data = geometry.shear(data, visualize=False, shear_factor=(0.1,0.3))
# data = geometry.rotate(data, visualize=False, degrees=35.8, center = center)
# data = transforms.change_brigntness(data, visualize=False, brigth=0.8)
# data = color.change_contrast(data, visualize=False, contrast=0.1)
# data = color.equalize_histogram(data, visualize=False)
# data = color.change_gamma(data, gamma=1.5, visualize=False)
# data = color.gaussian_blur(data)
# data = color.blur(data)
# data = color.inyect_salt_and_pepper_noise(data)
# data = color.inyect_spekle_noise(data)
# data = color.inyect_poisson_noise(data) (!!!!)
# data['image'] = color.inyect_gaussian_noise(data['image'], var=0.5)


"""


from operations import utils

utils.keypoints_to_homogeneus_and_concatenate(points)

#data_warped: torch.tensor = kornia.warp_affine(data.float(), M, dsize=(h, w))
#data_warped: torch.tensor = kornia.hflip(data_warped)
#data_warped: torch.tensor = geometry.scale(data, 0.8)


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
img_warped: np.ndarray = kornia.tensor_to_image(data_warped['image'].byte()[0])"""

# create the plot
img_warped: np.ndarray = kornia.tensor_to_image(data['image'].byte()[0])
fig, axs = plt.subplots(1, 2, figsize=(16, 10))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('image source')
xvalues = [value[0] for value in keypoints]
yvalues = [value[1] for value in keypoints]
axs[0].scatter(x=xvalues, y=yvalues, s=80)
axs[0].imshow(img)

axs[1].axis('off')
axs[1].set_title('image warped')
xvalues_warped = [value[0].cpu().numpy() for value in data['keypoints']]
yvalues_warped = [value[1].cpu().numpy() for value in data['keypoints']]
axs[1].scatter(x=xvalues_warped, y=yvalues_warped, s=80)
axs[1].imshow(img_warped)
plt.show()

"""
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


"""
