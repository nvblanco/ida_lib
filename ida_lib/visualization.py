import os
import sys

import cv2
import kornia
import numpy as np
import torch
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, Panel, Tabs, LabelSet, Div
from bokeh.models.widgets import CheckboxGroup
from bokeh.plotting import figure

__all__ = ['visualize', 'plot_image_transformation']

from ida_lib.operations.utils import get_principal_type

PLOT_SIZE = (550, 550)

color_palette = [
    # [R,G,B]
    [45, 171, 41],
    [41, 171, 130],
    [140, 63, 156],
    [196, 68, 24],
    [247, 255, 0],
    [0, 126, 11],
    [128, 185, 169],
    [0, 18, 130],
    [255, 0, 128],
    [88, 175, 98],
    [140, 140, 140],
    [153, 70, 70],
    [115, 255, 227],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255]
]

color_index = 0


# Changes the actual color index
def _get_next_color():
    global color_index
    color_index = (color_index + 1) % len(color_palette)
    return color_palette[color_index]


# Restart the color palette (to use the same colors in each item)
def _restart_color_palette():
    global color_index
    color_index = 0


def _get_image_types(image: dict):
    heatmap_labels = []
    image_labels = []
    mask_types = []
    points_types = []
    other_types = []
    for label in image.keys():
        if 'image' in label:
            image_labels.append(label)
        elif 'mask' in label or 'segmap' in label:
            mask_types.append(label)
        elif 'heatmap' in label:
            heatmap_labels.append(label)
        elif 'keypoints' in label:
            points_types.append(label)
        else:
            other_types.append(label)
    return image_labels, heatmap_labels, mask_types, points_types, other_types


# Process the input image and returned a ColumnDataSource with the image info to display it
def _process_image(img_orig):
    img = img_orig.copy().astype(np.uint8)
    if img.ndim == 2:  # gray input
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    elif img.ndim == 3:  # rgb input
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img = np.flipud(img)
    source = ColumnDataSource(data=dict(
        image=[img], x=[0], y=[img.shape[0]],
        dw=[img.shape[1]], dh=[img.shape[0]],
        R=[img[::-1, :, 0]], G=[img[::-1, :, 1]], B=[img[::-1, :, 2]]))
    return source


# Process the input mask and returned a ColumnDataSource with the mask info to display it
def _process_mask(img_orig):
    img = img_orig.copy().astype(np.uint8)
    img = np.flipud(img)
    source = ColumnDataSource(data=dict(
        image=[img], x=[0], y=[img.shape[0]],
        dw=[img.shape[1]], dh=[img.shape[0]],
        R=[img[::-1, :, 0]], G=[img[::-1, :, 1]], B=[img[::-1, :, 2]]))
    return source


def _process_points(points):
    if type(points) is np.ndarray:
        xvalues_warped = points[:, 0]
        yvalues_warped = points[:, 1]
    else:
        xvalues_warped = [(value[0].cpu().numpy()).astype(int) for value in points]
        yvalues_warped = [(value[1].cpu().numpy()).astype(int) for value in points]
    source = ColumnDataSource(data=dict(height=yvalues_warped,
                                        weight=xvalues_warped,
                                        names=range(xvalues_warped.__len__())))
    return xvalues_warped, yvalues_warped, source


def _generate_image_plot(img, tittle):
    if torch.is_tensor(img):
        img = img.to('cpu')
        img = kornia.tensor_to_image(img.byte())
    aspect = img.shape[0] / img.shape[1]
    img = _process_image(img)

    plot = figure(title=tittle, x_range=(0, img.data['dw'][0]), y_range=(
        img.data['dh'][0], 0), plot_width=PLOT_SIZE[0], plot_height=int(PLOT_SIZE[1] * aspect))
    plot.title.text_font_size = '12pt'
    plot.title.text_font_style = 'normal'
    plot.title.text_font = 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif'
    plot.image_rgba(source=img, image='image', x='x', y='y', dw='dw', dh='dh')
    return plot


def _add_mask_plot_and_checkbox(img, img2, color, mask, plot, plot2):
    if torch.is_tensor(img):
        img = img.to('cpu')
        img = torch.cat((img * color[0], img * color[1], img * color[2], np.clip(img * 204, 0, 204)), 0)
        img = kornia.tensor_to_image(img.byte())
    img1 = _process_mask(img)

    img2 = np.concatenate((img2 * color[0], img2 * color[1], img2 * color[2], np.clip(img2 * 204, 0, 204)), 2)

    if torch.is_tensor(img2):
        img2 = img2.to('cpu')
        img2 = kornia.tensor_to_image(img2.byte())
    img2 = _process_mask(img2)

    img_plot = plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')  # transformed
    img_plot2 = plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')  # original
    checkboxes_mask = CheckboxGroup(labels=[mask], active=[0, 1])
    callback_mask = CustomJS(code="""points.visible = false; 
                                     points_2.visible = false;
                                     if (cb_obj.active.includes(0)){points.visible = true;} 
                                     if (cb_obj.active.includes(0)){points_2.visible = true;}""",
                             args={'points': img_plot, 'points_2': img_plot2})
    checkboxes_mask.js_on_click(callback_mask)
    return checkboxes_mask


def _add_points_plot(points1, points2, plot, plot2):
    xval1, yval1, source = _process_points(points1)
    xval2, yval2, source2 = _process_points(points2)
    points = plot.circle(xval1, yval1, size=10, color="navy", alpha=0.8)
    if isinstance(xval2, np.ndarray):
        xval2 = xval2.transpose().tolist()
    if isinstance(yval2, np.ndarray):
        yval2 = yval2.transpose().tolist()
    points_2 = plot2.circle(xval2, yval2, size=10, color="navy", alpha=0.8)
    labels = LabelSet(x='weight', y='height', text='names', source=source,
                      x_offset=5, y_offset=5, render_mode='canvas')
    labels2 = LabelSet(x='weight', y='height', text='names', source=source2,
                       x_offset=5, y_offset=5, render_mode='canvas')
    plot.add_layout(labels)
    plot2.add_layout(labels2)

    checkboxes = CheckboxGroup(labels=['points'], active=[0, 1])
    callback = CustomJS(code="""points.visible = false; 
                                            labels.visible = false;
                                            labels2.visible = false;
                                            points_2.visible = false;
                                            if (cb_obj.active.includes(0)){points.visible = true;} 
                                            if (cb_obj.active.includes(0)){labels.visible = true;}
                                            if (cb_obj.active.includes(0)){labels2.visible = true;}
                                            if (cb_obj.active.includes(0)){points_2.visible = true;}""",
                        args={'points': points, 'labels': labels, 'labels2': labels2, 'points_2': points_2})
    checkboxes.js_on_click(callback)
    return checkboxes


def _add_label_plot(label):
    data_label = label
    if not isinstance(label, str):
        data_label = str(label)
    html = "<div style='padding: 5px; border-radius: 3px; background-color: #8ebf42'>\
    <span style='color:black;font-size:130%'>" + label + ":</span> " + data_label + '</div>'
    return Div(text=html,
               style={'font-family': 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif',
                      'font-size': '100%', 'color': '#17705E'})


def _generate_icon():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    icon_img: np.ndarray = cv2.imread(os.path.join(root_dir, 'static/icon2.png'))
    icon_img = cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB)
    icon_img = _process_image(icon_img)
    icon = figure(x_range=(0, icon_img.data['dw'][0]), y_range=(
        icon_img.data['dh'][0], 0), plot_width=130, plot_height=130)
    icon.margin = (25, 50, 20, 20)

    icon.xaxis.visible = False
    icon.yaxis.visible = False
    icon.min_border = 0
    icon.toolbar.logo = None
    icon.toolbar_location = None
    icon.xgrid.visible = False
    icon.ygrid.visible = False
    icon.outline_line_color = None
    icon.image_rgba(source=icon_img, image='image', x='x', y='y', dw='dw', dh='dh')
    return icon


def generate_title_template(template: int = 0):
    icon = _generate_icon()
    pre = Div(text="<b><div style='color:#224a42;font-size:280%' >IDALIB.</div>   image data augmentation</b>",
              style={'font-family': 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif', 'font-size': '250%',
                     'width': '5000', 'color': '#17705E'})
    pre.width = 1000
    if template == 0:
        description = Div(
            text="<b>This is the visualization tool of the IDALIB image data augmentation library. The first 5 samples \
            of the image batch are shown with the corresponding pipeline transformations. You can select to make \
            visible or not each of the data elements in the right column</b>",
            style={'font-family': 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif', 'font-size': '100%',
                   'font-weight': 'lighter', 'width': '5000', 'color': '#939393'})
    else:
        description = Div(
            text="<b>This is the visualization tool of the IDALIB image data augmentation library. Yo can see the \
            original image and the result image. You can select to make visible or not each of the data elements in \
            the right column</b>",
            style={'font-family': 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif', 'font-size': '100%',
                   'font-weight': 'lighter', 'width': '5000', 'color': '#939393'})
    title = column(row(icon, pre), description)
    title.margin = (0, 0, 20, 20)
    return title


def generate_item_tab(data, data_original, heatmap_labels, mask_types, points_types, other_types):
    list_target = ()
    list_checkbox = ()
    ppal_type = get_principal_type(data)
    img = data[ppal_type]
    plot = _generate_image_plot(img, 'transformed image')  # Generate plot of transformed image
    img2 = data_original[ppal_type]
    plot2 = _generate_image_plot(img2, 'original_image')  # Generate plot of original image
    list_plots = (plot2, plot)  # Add plots to the list of output plots
    for mask in mask_types:  # Loop through mask types of the input element
        img = data[mask]
        img2 = data_original[mask]
        color = _get_next_color()
        checkboxes_mask = _add_mask_plot_and_checkbox(img, img2, color, mask, plot, plot2)
        list_checkbox = (*list_checkbox, checkboxes_mask)
    for heatmap in heatmap_labels:  # Plotting of heatmap
        img = data[heatmap]
        img2 = data_original[heatmap]
        color = _get_next_color()
        checkboxes_heatmap = _add_mask_plot_and_checkbox(img, img2, color, heatmap, plot, plot2)
        list_checkbox = (*list_checkbox, checkboxes_heatmap)
    for keypoints in points_types:  # Plotting of keypoints
        points = data[keypoints]
        points2 = data_original[keypoints]
        checkboxes = _add_points_plot(points, points2, plot, plot2)
        list_checkbox = (*list_checkbox, checkboxes)
    for label in other_types:  # Plotting of labels
        data_label = data[label]
        label = _add_label_plot(data_label)
        list_target = (*list_target, label)
    # Configure tab elements
    if len(list_target) != 0:
        title_targets = Div(text="<b>Targets</b><hr>",
                            style={'font-family': 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif',
                                   'font-size': '100%', 'color': '#bf6800'})
        list_checkbox = (*list_checkbox, title_targets, *list_target)

    pre = Div(text="<b>Data Elements </b><hr>",
              style={'font-family': 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif', 'font-size': '150%',
                     'color': '#17705E'})
    checkboxes_column = column(pre, *list_checkbox)
    vertical_line = Div(text="<div></div>",
                        style={'border-right': '1px solid #e5e5e5', 'height': '100%', 'width': '30px'})
    list_plots = (*list_plots, vertical_line, checkboxes_column)
    p = row(*list_plots)
    return p


def visualize(images: dict, images_originals: dict, max_images: int = 5):
    """
    Generate the bokeh plot of the input batch transformation
    :param images: list of transformed items (dict of image and other  objects)
    :param images_originals: list of original items (dict of image and other  objects)
    :param max_images: max number of tabs to be shown
    """
    tabs = []
    image_labels, heatmap_labels, mask_types, points_types, other_types = _get_image_types(images[0])
    # loop through the input elements to create the tabs
    for index, (data, data_original) in enumerate(zip(images, images_originals)):
        # Restart palette of colors to have the same colors in each image
        if index == max_images:
            break
        _restart_color_palette()

        p = generate_item_tab(data=data, data_original=data_original, heatmap_labels=heatmap_labels,
                              mask_types=mask_types, points_types=points_types, other_types=other_types)
        title = 'image ' + str(index)
        tab = Panel(child=p, title=title)
        tabs.append(tab)

    # Generate output document
    layout = Tabs(tabs=tabs)
    title = generate_title_template()
    layout = column(title, layout)
    curdoc().title = "Batch visualization"
    curdoc().add_root(layout)

    # Run  bokeh server to show the visualization window
    command = 'bokeh serve --show ' + sys.argv[0]
    os.system(command)


def plot_image_transformation(data, data_original):
    """
        Generate the bokeh plot of the input batch transformation
        :param data: input dict element
        :param data_original: original input element (before transforms)
        """

    image_labels, heatmap_labels, mask_types, points_types, other_types = _get_image_types(data)
    layout = generate_item_tab(data, data_original, heatmap_labels, mask_types, points_types, other_types)

    title = generate_title_template(1)
    layout = column(title, layout)
    curdoc().title = "Batch visualization"
    curdoc().add_root(layout)

    # Run  bokeh server to show the visualization window
    command = 'bokeh serve --show ' + sys.argv[0]
    os.system(command)
