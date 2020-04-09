from bokeh.plotting import figure, output_file, show
from bokeh.models import Panel, Tabs
from bokeh.layouts import row
import cv2
import numpy as np
import torch
import kornia
import bokeh.plotting
from operations import geometry



def process_image(img_orig):
    img = img_orig.copy().astype(np.uint8)
    if img.ndim == 2:  # gray input
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.ndim == 3:  # rgb input
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img = np.flipud(img)
    source = bokeh.plotting.ColumnDataSource(data=dict(
        img=[img], x=[0], y=[img.shape[0]],
        dw=[img.shape[1]], dh=[img.shape[0]],
        R=[img[::-1, :, 0]], G=[img[::-1, :, 1]], B=[img[::-1, :, 2]]))
    return source

def generate_tab(image, keypoints,mask=None,  title=None,  **figure_kwargs):
    image = kornia.tensor_to_image(image.byte()[0])
    source = process_image(image)

    p1 = bokeh.plotting.figure(title=title, x_range=(0, img.shape[1]), y_range=(
        img.shape[0], 0), tools="pan,box_select,wheel_zoom", **figure_kwargs)
    p1.add_tools(bokeh.models.HoverTool(
        tooltips=[
            ("(x, y)", "($x, $y)"),
            ("RGB", "(@R, @G, @B)")]))

    p1.image_rgba(source=source, image='img', x='x', y='y', dw='dw', dh='dh')
    tab1_original = Panel(child=p1, title="image")

    p2 = bokeh.plotting.figure(title=title,  x_range=(0, img.shape[1]), y_range=(
        img.shape[0], 0), tools="pan,box_select,wheel_zoom", **figure_kwargs)
    p2.add_tools(bokeh.models.HoverTool(
        tooltips=[
            ("(x, y)", "($x, $y)"),
            ("RGB", "(@R, @G, @B)")]))
    p2.image_rgba(source=source, image='img', x='x', y='y', dw='dw', dh='dh')

    xvalues_warped = [(value[0].numpy()).astype(int) for value in keypoints]
    yvalues_warped = [(value[1].numpy()).astype(int) for value in keypoints]

    # xvalues_warped = [128 ,23]
    # yvalues_warped = [128, 78]
    p2.circle(xvalues_warped, yvalues_warped, size=10, color="navy", alpha=0.5)
    tab2_original = Panel(child=p2, title="keypoints")

    if mask is not None:
        p3 = bokeh.plotting.figure(title=title, x_range=(0, img.shape[1]), y_range=(
            img.shape[0], 0), tools="pan,box_select,wheel_zoom", **figure_kwargs)
        p3.add_tools(bokeh.models.HoverTool(
            tooltips=[
                ("(x, y)", "($x, $y)"),
                ("RGB", "(@R, @G, @B)")]))
        p3.image_rgba(source=source, image='img', x='x', y='y', dw='dw', dh='dh')
        mask = kornia.tensor_to_image(mask.byte()[0])
        source_mask = process_image(mask)
        p3.image_rgba(source=source_mask, image='img', x='x', y='y', dw='dw', dh='dh')
        tab3_original = Panel(child=p3, title="mask")
        tabs_original = Tabs(tabs=[tab1_original, tab2_original, tab3_original])
    else:
        tabs_original = Tabs(tabs=[tab1_original, tab2_original])
    return tabs_original

def bokeh_imshow(data, **figure_kwargs):


    bokeh.plotting.output_file("data_visualization.html")

    tabs_original = generate_tab(data['original']['image'], data['original']['keypoints'],mask = data['original']['mask'],  title='original_image')
    tabs_warped = generate_tab(data['image'], data['keypoints'], mask = data['mask'], title='warped_image')
    bokeh.plotting.show(row(tabs_original, tabs_warped))  # open a browser

import numpy

from PIL import Image
from bokeh.plotting import figure, show


width = 500
height = 500

# read the image with OpenCV
img: np.ndarray = cv2.imread('../gato.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

keypoints = ([img.shape[0]//2, img.shape[1]//2], [img.shape[0]//2  + 105, img.shape[1]//2 - 50])

# convert to torch tensor
data: torch.tensor = kornia.image_to_tensor(img, keepdim=False)  # BxCxHxW
points = [torch.from_numpy(np.asarray(point)) for point in keypoints]

data = {'image':data, 'mask': data,  'keypoints': points}
#input data

data = geometry.hflip(data, visualize=True)

bokeh_imshow(data)

'''
# Display the 32-bit RGBA image
dim = max(xdim, ydim)
fig = figure(title="Lena",
             x_range=(0,dim), y_range=(0,dim),
             # Specifying xdim/ydim isn't quire right :-(
             # width=xdim, height=ydim,
             )
fig.image_rgba(image=[img], x=0, y=0, dw=xdim, dh=ydim)

output_file("lena.html", title="image example")

show(fig)  # open a browser


N1 = img_warped.shape[0]
N2 = img_warped.shape[1]

img = np.zeros((N1,N2), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((N1, N2, 4))

view[:N1,:N2,0] = img[range(N1-1,-1,-1),:N2,0]
view[:N1,:N2,1] = img[range(N1-1,-1,-1),:N2,1]
view[:N1,:N2,2] = img[range(N1-1,-1,-1),:N2,2]

output_file("image_visualization.html")


#Original image
p1 = figure(title="original image",tools="pan,lasso_select,box_select,wheel_zoom",  x_range=(0,width), y_range=(0,height))
p1.image_rgba(image=[img], x=0, y=0, dw=width, dh=height, level="image")
#p1.image_url(url=['./gato.jpg'], x=0, y=height, w=width, h=height)
tab1_original = Panel(child=p1, title="image")

p2 = figure(title="original image",tools="pan,lasso_select,box_select,wheel_zoom", x_range=(0,width), y_range=(0,height))
p2.image_url(url=['./gato.jpg'], x=0, y=height, w=width, h=height)
p2.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=width, color="navy", alpha=0.5)
tab2_original = Panel(child=p2, title="keypoints")

# show the results
tabs_original = Tabs(tabs=[ tab1_original, tab2_original ])

#Warped image
p1_w = figure(title="warped image",tools="pan,lasso_select,box_select,wheel_zoom",  x_range=(0,width), y_range=(0,height))
p1_w.image_url(url=['./gato.jpg'], x=0, y=height, w=width, h=height)
tab1_warped = Panel(child=p1_w, title="image")

p2_w = figure(title="warped image",tools="pan,lasso_select,box_select,wheel_zoom",  x_range=(0,width), y_range=(0,height))
p2_w.image_url(url=['./gato.jpg'], x=0, y=height, w=width, h=height)
p2_w.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=width, color="navy", alpha=0.5)
tab2_warped = Panel(child=p2_w, title="keypoints")

# show the results
tabs_warped = Tabs(tabs=[ tab1_warped, tab2_warped])

show(row(tabs_original, tabs_warped))
'''