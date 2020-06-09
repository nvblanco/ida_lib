import bokeh.plotting
import cv2
import kornia
import numpy as np
import torch
from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.models import Panel
from bokeh.models import Select, ColumnDataSource, CustomJS, Panel, Tabs
from bokeh.models import ranges, LabelSet
from bokeh.models.widgets import CheckboxGroup
from bokeh.palettes import PuBu
from bokeh.plotting import figure


def diference_between_images_pixel(img1, img2):
    if img1.shape != img2.shape:
        raise Exception("Images must have the same dimensions to compare")
    #if type(img1).__module__ != np.__name__ or type(img1).__module__ != np.__name__ :
    #    raise Exception("Images must be numpy array")
    """if img2.device == torch.device('cpu') : img2 = img2.to('cuda')
    if img1.device == torch.device('cpu'): img1 = img1.to('cuda')"""
    dif = torch.abs(torch.sub(img1[0, 0, ...], img2[0, 0, ...]))
    dif = dif+torch.abs(torch.sub(img1[0, 1, ...], img2[0, 1, ...]))
    dif = dif+torch.abs(torch.sub(img1[0, 2, ...], img2[0, 2, ...]))
    dif = dif / 3

    dif_total = torch.sum(torch.sum(dif, axis=0), axis=0)

    _,w, h = img1.shape
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
    image = kornia.tensor_to_image(image.byte())
    source = process_image(image)

    p1 = bokeh.plotting.figure(title=title, x_range=(0, image.shape[1]), y_range=(
        image.shape[0], 0), tools="pan,box_select,wheel_zoom", **figure_kwargs)
    #p1.title.text_font_size = "25px"
    p1.add_tools(bokeh.models.HoverTool(
        tooltips=[
            ("(x, y)", "($x, $y)"),
            ("RGB", "(@R, @G, @B)")]))

    p1.image_rgba(source=source, image='img', x='x', y='y', dw='dw', dh='dh')
    tab1_original = Panel(child=p1, title="image")

    p2 = bokeh.plotting.figure(title=title,  x_range=(0, image.shape[1]), y_range=(
        image.shape[0], 0), tools="pan,box_select,wheel_zoom", **figure_kwargs)
    p2.add_tools(bokeh.models.HoverTool(
        tooltips=[
            ("(x, y)", "($x, $y)"),
            ("RGB", "(@R, @G, @B)")]))
    p2.image_rgba(source=source, image='img', x='x', y='y', dw='dw', dh='dh')
    tabs = [tab1_original]
    if  keypoints is not None:
        xvalues_warped = [(value[0].cpu().numpy()).astype(int) for value in keypoints]
        yvalues_warped = [(value[1].cpu().numpy()).astype(int) for value in keypoints]
        labels = range(xvalues_warped.__len__())

        source = ColumnDataSource(data=dict(height=yvalues_warped,
                                            weight=xvalues_warped,
                                            names=labels))

        p2.circle(xvalues_warped, yvalues_warped, size=10, color="navy", alpha=0.5)
        labels = LabelSet(x='weight', y='height', text='names',source=source,
                          x_offset=5, y_offset=5, render_mode='canvas')
        p2.add_layout(labels)
        tab2_original = Panel(child=p2, title="keypoints")
        tabs.append(tab2_original)
    if mask is not None:
        mask = kornia.tensor_to_image(mask.byte()[0])
        p3 = bokeh.plotting.figure(title=title, x_range=(0, image.shape[1]), y_range=(
            image.shape[0], 0), tools="pan,box_select,wheel_zoom", **figure_kwargs)
        source_mask = process_image(mask)
        p3.image_rgba(source=source_mask, image='img', x='x', y='y', dw='dw', dh='dh')
        """p3.add_tools(bokeh.models.HoverTool(
            tooltips=[
                ("(x, y)", "($x, $y)"),
                ("RGB", "(@R, @G, @B)")]))
        p3.image_rgba(source=source, image='img', x='x', y='y', dw='dw', dh='dh')
        mask = kornia.tensor_to_image(mask.byte()[0])
        #source_mask = process_image(mask)
        #p3.image_rgba(source=source_mask, image='img', x='x', y='y', dw='dw', dh='dh')"""
        tab3_original = Panel(child=p3, title="mask")
        tabs.append(tab3_original)
    tabs_original = Tabs(tabs=tabs)
    return tabs_original

def plot_image_tranformation(data, data_original, **figure_kwargs):
    if not torch.is_tensor(data['image']): data['image'] = (kornia.image_to_tensor(data['image'])).squeeze()
    if not torch.is_tensor(data_original['image']): data_original['image'] = (kornia.image_to_tensor(data_original['image'])).squeeze()
    dif = diference_between_images_pixel(data['image'].squeeze(), data_original['image'].squeeze())
    bokeh.plotting.output_file("data_visualization.html")

    if data_original.keys().__contains__('mask'):
        mask_original = data_original['mask']
    else:
        mask_original = None
    if data.keys().__contains__('mask'):
        mask = data['mask']
    else:
        mask = None

    if data_original.keys().__contains__('keypoints'):
        keypoints_original = data_original['keypoints']
    else:
        keypoints_original = None
    if data.keys().__contains__('keypoints'):
        keypoints = data['keypoints']
    else:
        keypoints = None


    tabs_original = generate_tab(data_original['image'], keypoints=keypoints_original,mask = mask_original,  title='original_image')
    tabs_warped = generate_tab(data['image'], keypoints=keypoints, mask = mask, title='warped_image')
    source = ColumnDataSource(dict(x=['Diference percentage'], y=[dif.numpy()]))

    title = "Image diference pixel-values"
    plot = figure(plot_width=200, plot_height=400, tools="",
                  title=title,
                  x_minor_ticks=2,
                  x_range=source.data["x"],
                  y_range=ranges.Range1d(start=0, end=100))

    labels = LabelSet(x='x', y='y', text='y', level='glyph',
                      x_offset=-13.5, y_offset=0, source=source, render_mode='canvas')

    plot.vbar(source=source, x='x', top='y', bottom=0, width=0.3, color=PuBu[7][2])
    plot.margin = (205, 5, 15, 5)
    plot.add_layout(labels)
    p = row(tabs_original, tabs_warped, plot)
    p.margin= (50,0,0,100)
    bokeh.plotting.show(p)  # open a browser

def generate_image_tab(img, img2, img_number):
    if torch.is_tensor(img):img = kornia.tensor_to_image(img.byte())
    img = process_image(img)
    if torch.is_tensor(img2):img2 = kornia.tensor_to_image(img2.byte())
    img2 = process_image(img2)
    plot = figure(title="holi", x_range=(0, img.data['dw'][0]), y_range=(
        img.data['dh'][0], 0))
    plot.image_rgba(source=img, image='image', x='x', y='y', dw='dw', dh='dh')
    points = plot.circle([100, 90, 180], [100, 90, 180], size=10, color="navy", alpha=0.5)
    points2 = plot.circle([150, 96, 180], [150, 60, 380], size=10, color="red", alpha=0.5)

    plot2 = figure(title="holi", x_range=(0, img2.data['dw'][0]), y_range=(
        img2.data['dh'][0], 0))
    plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')
    points_2 = plot2.circle([100, 90, 180], [100, 90, 180], size=10, color="navy", alpha=0.5)
    points2_2 = plot2.circle([150, 96, 180], [150, 60, 380], size=10, color="red", alpha=0.5)

    select_widget = Select(options=['uniform', 'normal'], value='uniform distribution', title='selec your choice')
    checkboxes = CheckboxGroup(labels=list(('points1', 'points2')), active=[0, 1])
    callback = CustomJS(code="""points.visible = false; // same xline passed in from args
                                    points_2.visible = false;
                                    points2_2.visible = false;
                                    points2.visible = false;
                                    // cb_obj injected in by the callback
                                    if (cb_obj.active.includes(0)){points.visible = true;} // 0 index box is xline
                                    if (cb_obj.active.includes(0)){points_2.visible = true;}
                                    if (cb_obj.active.includes(1)){points2_2.visible = true;}
                                    if (cb_obj.active.includes(1)){points2.visible  = true;}""",
                        args={'points': points, 'points2': points2, 'points_2': points_2, 'points2_2': points2_2})
    checkboxes.js_on_click(callback)
    p = row(plot, plot2, checkboxes)
    return Panel(child=p, title="image1")


def plot_batch_image_tranformation(data, data_original, **figure_kwargs):
    data = data[3]
    data['image']=data['image'].to('cpu')
    data_original = data_original[3]
    if not torch.is_tensor(data['image']): data['image'] = (kornia.image_to_tensor(data['image'])).squeeze()
    if not torch.is_tensor(data_original['image']): data_original['image'] = (kornia.image_to_tensor(data_original['image'])).squeeze()
    dif = diference_between_images_pixel(data['image'].squeeze(), data_original['image'].squeeze())
    bokeh.plotting.output_file("data_visualization.html")

    t1 = generate_image_tab(data['image'], data_original['image'], 5)
    t2 = generate_image_tab(data['image'], data_original['image'], 5)
    layout = Tabs(tabs=[t1, t2])
    curdoc().title = "Batch visualization"
    curdoc().add_root(layout)

    import os

    os.system('bokeh serve --show visualization.py')


def plot_batch_image_tranformation_old(data, data_original, **figure_kwargs):
    data = data[3]
    data['image']=data['image'].to('cpu')
    data_original = data_original[3]
    if not torch.is_tensor(data['image']): data['image'] = (kornia.image_to_tensor(data['image'])).squeeze()
    if not torch.is_tensor(data_original['image']): data_original['image'] = (kornia.image_to_tensor(data_original['image'])).squeeze()
    dif = diference_between_images_pixel(data['image'].squeeze(), data_original['image'].squeeze())
    bokeh.plotting.output_file("data_visualization.html")

    if data_original.keys().__contains__('mask'):
        mask_original = data_original['mask']
    else:
        mask_original = None
    if data.keys().__contains__('mask'):
        mask = data['mask']
    else:
        mask = None

    if data_original.keys().__contains__('keypoints'):
        keypoints_original = data_original['keypoints']
    else:
        keypoints_original = None
    if data.keys().__contains__('keypoints'):
        keypoints = data['keypoints']
    else:
        keypoints = None


    tabs_original = generate_tab(data_original['image'], keypoints=keypoints_original,mask = mask_original,  title='original_image')
    tabs_warped = generate_tab(data['image'], keypoints=keypoints, mask = mask, title='warped_image')
    source = ColumnDataSource(dict(x=['Diference percentage'], y=[dif.numpy()]))

    title = "Image diference pixel-values"
    plot = figure(plot_width=200, plot_height=400, tools="",
                  title=title,
                  x_minor_ticks=2,
                  x_range=source.data["x"],
                  y_range=ranges.Range1d(start=0, end=100))

    labels = LabelSet(x='x', y='y', text='y', level='glyph',
                      x_offset=-13.5, y_offset=0, source=source, render_mode='canvas')

    plot.vbar(source=source, x='x', top='y', bottom=0, width=0.3, color=PuBu[7][2])
    plot.margin = (205, 5, 15, 5)
    plot.add_layout(labels)
    p = row(tabs_original, tabs_warped, plot)
    p.margin= (50,0,0,100)
    bokeh.plotting.show(p)  # open a browser
"""
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
"""

