from bokeh.plotting import figure, output_file, show
from bokeh.models import Panel, Tabs
from bokeh.layouts import row, column
import cv2
import numpy as np
import torch
import kornia
import bokeh.plotting
from bokeh.palettes import PuBu
from bokeh.models import ColumnDataSource, ranges, LabelSet
from operations import geometry
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, VBar, HBar


def diference_between_images_pixel(img1, img2):
    if img1.shape != img2.shape:
        raise Exception("Images must have the same dimensions to compare")
    #if type(img1).__module__ != np.__name__ or type(img1).__module__ != np.__name__ :
    #    raise Exception("Images must be numpy array")
    dif = torch.abs(torch.sub(img1[0, 0, ...], img2[0, 0, ...]))
    dif = dif+torch.abs(torch.sub(img1[0, 1, ...], img2[0, 1, ...]))
    dif = dif+torch.abs(torch.sub(img1[0, 2, ...], img2[0, 2, ...]))
    dif = dif / 3

    dif_total = torch.sum(torch.sum(dif, axis=0), axis=0)

    _,_,w, h = img1.shape
    total = w * h * 256
    diference = ( dif_total / total ) * 100
    return diference


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

def plot_image_tranformation(data, data_original, **figure_kwargs):
    dif = diference_between_images_pixel(data['image'], data['original']['image'])
    bokeh.plotting.output_file("data_visualization.html")

    tabs_original = generate_tab(data_original['image'], data_original['keypoints'],mask = data_original['mask'],  title='original_image')
    tabs_warped = generate_tab(data['image'], data['keypoints'], mask = data['mask'], title='warped_image')
    source = ColumnDataSource(dict(x=['Diference percentage'], y=[dif.numpy()]))

    title = "Image diference pixel-values"
    plot = figure(plot_width=300, plot_height=500, tools="",
                  title=title,
                  x_minor_ticks=2,
                  x_range=source.data["x"],
                  y_range=ranges.Range1d(start=0, end=100))

    labels = LabelSet(x='x', y='y', text='y', level='glyph',
                      x_offset=-13.5, y_offset=0, source=source, render_mode='canvas')

    plot.vbar(source=source, x='x', top='y', bottom=0, width=0.3, color=PuBu[7][2])

    plot.add_layout(labels)
    bokeh.plotting.show(row(tabs_original, tabs_warped, plot))  # open a browser


'''
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


center = torch.ones(1, 2)
center[...,0]=100
center[...,1]= 50

from operations import functional

def vflip(data, visualize = False):
    op = functional.vflip_class(data, visualize)
    return op()

data = vflip(data)
#data = geometry.hflip(data)
#data = geometry.rotate(data,degrees=35, visualize=True, center=center)

#plot_image_tranformation(data)
'''

