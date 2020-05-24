from bokeh.io.state import curstate
from bokeh.models import Button, ColumnDataSource, CustomJS, Panel, Tabs, LabelSet, PreText, Div
from bokeh.models.widgets import CheckboxGroup
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.plotting import figure
import numpy as np
import cv2
import torch
import kornia
import os
import sys


__all__ = [ 'visualize']


PLOT_SIZE = (550, 550)

color_palette = [
    #[R,G,B]
    [45,    171,    41],
    [41,    171,    130],
    [140,   63,     156],
    [196,   68,     24],
    [247,   255,    0],
    [0,     126,    11],
    [128,   185,    169],
    [0,     18,     130],
    [255,   0,      128],
    [88,    175,    98],
    [140,   140,    140],
    [153,   70,     70],
    [115,   255,    227],
    [255,   0,      0],
    [0,     255,    0],
    [0,     0,      255]
]

color_index = 0

def get_next_color():
    global color_index
    color_index = (color_index + 1 ) % len(color_palette)
    return color_palette[color_index]



#Process the input image and returned a ColumnDataSource wirh the image info to display it
def process_image(img_orig):
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


def process_mask(img_orig):
    img = img_orig.copy().astype(np.uint8)
    img = np.flipud(img)
    source = ColumnDataSource(data=dict(
        image=[img], x=[0], y=[img.shape[0]],
        dw=[img.shape[1]], dh=[img.shape[0]],
        R=[img[::-1, :, 0]], G=[img[::-1, :, 1]], B=[img[::-1, :, 2]]))
    return source


def visualize(images, images_originals, mask_types, other_types,  max_images = 5):
    tabs = []
    for index, (data, data_original) in enumerate(zip(images, images_originals)):
        #Restart palette of colors to have the same colors in each image
        global color_index
        color_index = 0
        list_target = ()
        list_checkbox = ()
        if index == max_images:
            break
        img = data['image']
        if torch.is_tensor(img):
            img = img.to('cpu')
            img = kornia.tensor_to_image(img.byte())
        aspect = img.shape [0]/img.shape [1]
        img1 = process_image(img)

        img2 = data_original['image']
        if torch.is_tensor(img2):
            img2 = img2.to('cpu')
            img2 = kornia.tensor_to_image(img2.byte())
        img2 = process_image(img2)

        plot = figure(title="transformed image", x_range=(0, img1.data['dw'][0]), y_range=(
            img1.data['dh'][0], 0), plot_width=PLOT_SIZE[0], plot_height=int(PLOT_SIZE[1]*aspect))
        plot.title.text_font_size = '12pt'
        plot.title.text_font_style = 'normal'
        plot.title.text_font = 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif'
        plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')

        plot2 = figure(title="original image", x_range=(0, img2.data['dw'][0]), y_range=(
            img2.data['dh'][0], 0), plot_width=PLOT_SIZE[0], plot_height=int(PLOT_SIZE[1]*aspect))
        plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')
        plot2.title.text_font_size = '12pt'
        plot2.title.text_font_style = 'normal'
        plot2.title.text_font = 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif'
        list_plots = (plot2, plot)
        for mask in mask_types:
            img = data[mask]
            color = get_next_color()
            if torch.is_tensor(img):
                img = img.to('cpu')
                one = torch.ones((1, img.shape[1], img.shape[2]))
                img = torch.cat((img * color[0] , img * color[1], img * color[2], np.clip(img * 204, 0, 204)), 0)
                img = kornia.tensor_to_image(img.byte())  
            img1 = process_mask(img)

            img2 = data_original[mask]
            one = np.ones((img2.shape[0], img2.shape[1], 1))
            img2 = np.concatenate((img2 * color[0], one * color[1], one * color[2], img2 * 204), 2)

            if torch.is_tensor(img2):
                img2 = img2.to('cpu')
                img2 = kornia.tensor_to_image(img2.byte())
            img2 = process_mask(img2)

            img_plot = plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')        #transformed
            img_plot2 = plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')      #original
            checkboxes_mask = CheckboxGroup(labels=[mask], active=[0, 1])
            callback_mask = CustomJS(code="""points.visible = false; 
                                             points_2.visible = false;
                                             if (cb_obj.active.includes(0)){points.visible = true;} 
                                             if (cb_obj.active.includes(0)){points_2.visible = true;}""",
                                args={'points': img_plot,  'points_2': img_plot2})
            checkboxes_mask.js_on_click(callback_mask)
            list_checkbox = (*list_checkbox, checkboxes_mask)
        if data.keys().__contains__('heatmap'):
            img = data['heatmap']
            if torch.is_tensor(img):
                img = img.to('cpu')
                img_256 = img * 255
                one = torch.ones((1, img.shape[1], img.shape[2]))
                #img = torch.cat((one * img_256 , one * (255-img_256), one * 25, img_256* 0.9 ), 0)
                img = kornia.tensor_to_image(img.byte())
                one = np.ones((img.shape[0], img.shape[1], 1))
                img_256 = img.reshape(img.shape[0], img.shape[1], 1)  * 255
                img = np.concatenate((one * img_256, one * (255 - img_256), one * 25, img_256 * 0.9), 2)
            img1 = process_mask(img)

            img2 = data_original['heatmap']
            if len(img2.shape) == 2:
                img2.reshape(img2.shape[0], img2.shape[1], 1)
            img2_256 = img2.reshape(img2.shape[0], img2.shape[1], 1) * 256
            one = np.ones((img2.shape[0], img2.shape[1], 1))
            img2 = np.concatenate((one * img2_256 , one * (255-img2_256), one * 25, img2_256* 0.9 ), 2)

            if torch.is_tensor(img2):
                img2 = img2.to('cpu')
                img2 = kornia.tensor_to_image(img2.byte())
            img2 = process_mask(img2)
            img_plot = plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')        #transformed
            img_plot2 = plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')      #original
            checkboxes_heatmap = CheckboxGroup(labels=['heatmap'], active=[0, 1])
            callback_heatmap = CustomJS(code="""points.visible = false; 
                                             points_2.visible = false;
                                             if (cb_obj.active.includes(0)){points.visible = true;} 
                                             if (cb_obj.active.includes(0)){points_2.visible = true;}""",
                                args={'points': img_plot,  'points_2': img_plot2})
            checkboxes_heatmap.js_on_click(callback_heatmap)
            list_checkbox = (*list_checkbox, checkboxes_heatmap)
        if data.keys().__contains__('keypoints'):
            points = data['keypoints']
            xvalues_warped = [(value[0].cpu().numpy()).astype(int) for value in points]
            yvalues_warped = [(value[1].cpu().numpy()).astype(int) for value in points]

            points2 = data_original['keypoints']
            if type(points2) is np.ndarray:
                xvalues_warped2 = points2[:,0]
                yvalues_warped2 = points2[:,1]
            else:
                xvalues_warped2 = [(value[0].cpu().numpy()).astype(int) for value in points2]
                yvalues_warped2 = [(value[1].cpu().numpy()).astype(int) for value in points2]
            source = ColumnDataSource(data=dict(height=yvalues_warped,
                                                weight=xvalues_warped,
                                                names=range(xvalues_warped.__len__())))

            points = plot.circle(xvalues_warped, yvalues_warped, size=10, color="navy", alpha=0.8)
            labels = LabelSet(x='weight', y='height', text='names', source=source,
                              x_offset=5, y_offset=5, render_mode='canvas')
            plot.add_layout(labels)


            source2 = ColumnDataSource(data=dict(height=yvalues_warped2,
                                                 weight=xvalues_warped2,
                                                 names=range(xvalues_warped2.__len__())))

            points_2 = plot2.circle(xvalues_warped2, yvalues_warped2, size=10, color="navy", alpha=0.8)
            labels2 = LabelSet(x='weight', y='height', text='names', source=source2,
                               x_offset=5, y_offset=5, render_mode='canvas')
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
            list_checkbox = (*list_checkbox, checkboxes)
        for label in other_types:
            data_label = data[label]
            if not isinstance(data[label], str): data_label = str(data[label])
            html = "<div style='padding: 5px; border-radius: 3px; background-color: #8ebf42'><span style='color:black;font-size:130%'>" + label + ":</span> " + data_label + '</div>'
            label = Div(text=html,
                      style={'font-family': 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif',
                             'font-size': '100%', 'color': '#17705E'})
            list_target = (*list_target, label)
        if len(list_target) != 0:
            title_targets = pre = Div(text="<b>Targets</b><hr>",
                      style={'font-family': 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif', 'font-size': '100%', 'color': '#bf6800'})
            list_checkbox = (*list_checkbox, title_targets, *list_target)
        title = 'image ' + str(index)
        pre = Div(text="<b>Data Elements </b><hr>",
                  style={'font-family': 'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif', 'font-size': '150%', 'color': '#17705E'})
        checkboxes_column = column(pre, *list_checkbox)
        vertical_line = Div(text="<div></div>",
                            style={'border-right': '1px solid #e5e5e5', 'height': '100%', 'width': '30px'})
        list_plots = (*list_plots, vertical_line, checkboxes_column)
        p = row(*list_plots)
        tab = Panel(child=p, title=title)
        tabs.append(tab)

    layout = Tabs(tabs=tabs)

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    icon_img: np.ndarray = cv2.imread(os.path.join(ROOT_DIR, 'icon2.png') )
    icon_img = cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB)
    icon_img = process_image(icon_img)
    icon = figure( x_range=(0, icon_img.data['dw'][0]), y_range=(
        icon_img.data['dh'][0], 0), plot_width=130, plot_height=130)
    icon.margin = (25, 50,20,20)

    icon.xaxis.visible = False
    icon.yaxis.visible = False
    icon.min_border = 0
    icon.toolbar.logo = None
    icon.toolbar_location = None
    icon.xgrid.visible = False
    icon.ygrid.visible = False
    icon.outline_line_color = None
    icon.image_rgba(source=icon_img, image='image', x='x', y='y', dw='dw', dh='dh')

    pre = Div(text="<b><div style='color:#224a42;font-size:280%' >IDALIB.</div>   image data augmentation</b>", style={'font-family':'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif', 'font-size': '250%', 'width':'5000', 'color': '#17705E'})
    pre.width = 1000
    description = Div(text="<b>This is the visualization tool of the IDALIB image data augmentation library. The first 5 samples of the image batch are shown with the corresponding pipeline transformations. You can select to make visible or not each of the data elements in the right column</b>", style={'font-family':'Avantgarde, TeX Gyre Adventor, URW Gothic L, sans-serif', 'font-size': '100%', 'font-weight': 'lighter', 'width':'5000', 'color': '#939393'})
    title = column(row(icon, pre), description)
    title.margin =(0,0,20,20)

    layout = column(title, layout)
    curdoc().title = "Batch visualization"
    curdoc().add_root(layout)

    command = 'bokeh serve --show ' + sys.argv[0]
    os.system(command)