from bokeh.models import Button, ColumnDataSource, CustomJS, Panel, Tabs, LabelSet
from bokeh.models.widgets import CheckboxGroup
from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.plotting import figure
import numpy as np
import cv2
import torch
import kornia



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
    img = np.concatenate((img, img, img), axis=2)
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


def visualize(images, images_originals, max_images = 5):
    tabs = []
    for index, (data, data_original) in enumerate(zip(images, images_originals)):
        if index == max_images:
            break
        img = data['image']
        if torch.is_tensor(img):
            img = img.to('cpu')
            img = kornia.tensor_to_image(img.byte())
        points = data['keypoints']
        xvalues_warped = [(value[0].cpu().numpy()).astype(int) for value in points]
        yvalues_warped = [(value[1].cpu().numpy()).astype(int) for value in points]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1 = process_image(img)

        img2 = data_original['image']
        if torch.is_tensor(img2):
            img2 = img2.to('cpu')
            img2 = kornia.tensor_to_image(img2.byte())
        points2 = data_original['keypoints']
        xvalues_warped2 = [(value[0].cpu().numpy()).astype(int) for value in points2]
        yvalues_warped2 = [(value[1].cpu().numpy()).astype(int) for value in points2]
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = process_image(img2)

        plot = figure(title="transformed image", x_range=(0, img1.data['dw'][0]), y_range=(
            img1.data['dh'][0], 0))
        plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')

        plot2 = figure(title="original image", x_range=(0, img2.data['dw'][0]), y_range=(
            img2.data['dh'][0], 0))
        plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')
        list_plots = (plot2, plot)
        if data.keys().__contains__('keypoints'):
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

            checkboxes = CheckboxGroup(labels=['points1'], active=[0, 1])
            callback = CustomJS(code="""points.visible = false; 
                                        labels.visible = false;
                                        labels2.visible = false;
                                        points_2.visible = false;
                                        if (cb_obj.active.includes(0)){points.visible = true;} 
                                        if (cb_obj.active.includes(0)){labels.visible = true;}
                                        if (cb_obj.active.includes(0)){labels2.visible = true;}
                                        if (cb_obj.active.includes(0)){points2.visible = true;}""",
                                args={'points': points, 'labels': labels, 'labels2': labels2, 'points_2': points_2})
            checkboxes.js_on_click(callback)
            list_plots = (*list_plots, checkboxes)
        if data.keys().__contains__('mask'):
            img = data['mask']
            if torch.is_tensor(img):
                img = img.to('cpu')
                img = (kornia.tensor_to_image(img.byte()))
                img = img.reshape(img.shape[0], img.shape[1], 1)
                img = np.concatenate((img, img, img), axis=2)

            #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img1 = process_image(img)

            img2 = data_original['mask']
            if torch.is_tensor(img2):
                img2 = img2.to('cpu')
                img2 = kornia.tensor_to_image(img2.byte())
            img2 = img2.reshape(img2.shape[0], img2.shape[1], 1)
            img2 = np.ones((img2.shape[0], img2.shape[1], 1)) * 256
                #img2 = np.concatenate((img2, img2, img2), axis=2)
            #img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
            img2 = process_image(img2)

            img_plot = plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')
            img_plot2 = plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')
            checkboxes_mask = CheckboxGroup(labels=['mask'], active=[0, 1])
            callback_mask = CustomJS(code="""points.visible = false; 
                                                    points_2.visible = false;
                                                    if (cb_obj.active.includes(0)){points.visible = true;} 
                                                    if (cb_obj.active.includes(0)){points2.visible = true;}""",
                                args={'points': img_plot,  'points_2': img_plot2})
            checkboxes_mask.js_on_click(callback_mask)
            list_plots = (*list_plots, checkboxes_mask)
        title = 'image ' + str(index)
        p = row(*list_plots)
        tab = Panel(child=p, title=title)
        tabs.append(tab)

    def button_callback():
        sys.exit()  # Stop the server

    button = Button(label="Stop", button_type="success")
    button.on_click(button_callback)

    layout = row(Tabs(tabs=tabs), button)
    curdoc().title = "Batch visualization"
    curdoc().add_root(layout)
    import os
    import sys
    command = 'bokeh serve --show ' + sys.argv[0]
    os.system(command)