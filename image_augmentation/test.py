from bokeh.io import curdoc
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Button, Select, ColumnDataSource, CustomJS, Panel, Tabs, LabelSet
from bokeh.models.widgets import CheckboxGroup
from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.plotting import figure
from numpy.random import random, normal
import numpy as np
import cv2
import torch
import kornia

tabs = []


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


def generate_image_tab(img1, img2, img_number):
    plot = figure(title="holi", x_range=(0, img1.data['dw'][0]), y_range=(
        img1.data['dh'][0], 0))
    plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')
    points = plot.circle([100, 90, 180], [100, 90, 180], size=10, color="navy", alpha=0.5)
    points2 = plot.circle([150, 96, 180], [150, 60, 380], size=10, color="red", alpha=0.5)

    plot2 = figure(title="holi", x_range=(0, img2.data['dw'][0]), y_range=(
        img2.data['dh'][0], 0))
    plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')
    points_2 = plot2.circle([100, 90, 180], [100, 90, 180], size=10, color="navy", alpha=0.5)
    points2_2 = plot2.circle([150, 96, 180], [150, 60, 380], size=10, color="red", alpha=0.5)

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


def add_tab(img, img2):
    plot = figure(title="holi", x_range=(0, img.data['dw'][0]), y_range=(
        img.data['dh'][0], 0))
    plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')
    points = plot.circle([100, 90, 180], [100, 90, 180], size=10, color="navy", alpha=0.5)
    points2 = plot.circle([150, 96, 180], [150, 60, 380], size=10, color="red", alpha=0.5)
    plot.diamond(x='x', y='y', source=data_points, color='red')

    plot2 = figure(title="holi", x_range=(0, img2.data['dw'][0]), y_range=(
        img2.data['dh'][0], 0))
    plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')
    points_2 = plot2.circle([100, 90, 180], [100, 90, 180], size=10, color="navy", alpha=0.5)
    points2_2 = plot2.circle([150, 96, 180], [150, 60, 380], size=10, color="red", alpha=0.5)
    plot2.diamond(x='x', y='y', source=data_points, color='red')

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
    tabs.append(Panel(child=p, title="image n"))
    '''layout = Tabs(tabs=[t1, t2])
    curdoc().title = "Batch visualization"
    curdoc().add_root(layout)

    import os

    os.system('bokeh serve --show main.py')
'''


def visualize(images, images_originals):
    tabs = []
    for index, (data, data_original) in enumerate(zip(images, images_originals)):
        if index == 5:
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
        p = row(plot2, plot)
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

            checkboxes = CheckboxGroup(labels=list(('points1', 'points2')), active=[0, 1])
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
            p = row(plot2, plot, checkboxes)
        title = 'image ' + str(index)
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


''' img = data['image']
 points = data['keypoints']
 xvalues_warped = [(value[0].cpu().numpy()).astype(int) for value in points]
 yvalues_warped = [(value[1].cpu().numpy()).astype(int) for value in points]

 if torch.is_tensor(img):
     img = img.to('cpu')
     img = kornia.tensor_to_image(img.byte())
 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 img1 = process_image(img)
 plot = figure(title="holi", x_range=(0, img1.data['dw'][0]), y_range=(
     img1.data['dh'][0], 0))
 plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')
 source = ColumnDataSource(data=dict(height=yvalues_warped,
                                     weight=xvalues_warped,
                                     names=range(xvalues_warped.__len__())))

 points = plot.circle(xvalues_warped, yvalues_warped, size=10, color="navy", alpha=0.9)'''


def visualize_test(data, data_original):
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

    plot = figure(title="holi", x_range=(0, img1.data['dw'][0]), y_range=(
        img1.data['dh'][0], 0))
    plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')
    source = ColumnDataSource(data=dict(height=yvalues_warped,
                                        weight=xvalues_warped,
                                        names=range(xvalues_warped.__len__())))

    points = plot.circle(xvalues_warped, yvalues_warped, size=10, color="navy", alpha=0.8)
    labels = LabelSet(x='weight', y='height', text='names', source=source,
                      x_offset=5, y_offset=5, render_mode='canvas')
    plot.add_layout(labels)

    plot2 = figure(title="holi", x_range=(0, img2.data['dw'][0]), y_range=(
        img2.data['dh'][0], 0))
    plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')
    source2 = ColumnDataSource(data=dict(height=yvalues_warped2,
                                         weight=xvalues_warped2,
                                         names=range(xvalues_warped2.__len__())))

    points_2 = plot2.circle(xvalues_warped2, yvalues_warped2, size=10, color="navy", alpha=0.8)
    labels2 = LabelSet(x='weight', y='height', text='names', source=source2,
                       x_offset=5, y_offset=5, render_mode='canvas')
    plot2.add_layout(labels2)

    checkboxes = CheckboxGroup(labels=list(('points1', 'points2')), active=[0, 1])
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
    p = row(plot, plot2, checkboxes)
    title = 'image ' + str(0)
    tab = Panel(child=p, title=title)
    tabs.append(tab)

    def button_callback():
        sys.exit()  # Stop the server

    button = Button(label="Stop", button_type="success")
    button.on_click(button_callback)
    layout = row(Tabs(tabs=tabs), button)
    p = row(p)
    curdoc().title = "Batch visualization"
    curdoc().add_root(layout)
    import os
    import sys
    command = 'bokeh serve --show ' + sys.argv[0]
    os.system(command)


'''img: np.ndarray = cv2.imread('../gato.jpg')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1 = process_image(img1)

img2: np.ndarray = cv2.imread('../bird.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = process_image(img2)

initial_points = 500
data_points = ColumnDataSource(data={'x': random(initial_points), 'y': random(initial_points)})

# data_image = ColumnDataSource( data = {'x': img, 'y':img2})

plot = figure(title="holi", x_range=(0, img.shape[1]), y_range=(
    img.shape[0], 0))
plot.image_rgba(source=img1, image='image', x='x', y='y', dw='dw', dh='dh')
points = plot.circle([100, 90, 180], [100, 90, 180], size=10, color="navy", alpha=0.5)
points2 = plot.circle([150, 96, 180], [150, 60, 380], size=10, color="red", alpha=0.5)
plot.diamond(x='x', y='y', source=data_points, color='red')

plot2 = figure(title="holi", x_range=(0, img.shape[1]), y_range=(
    img.shape[0], 0))
plot2.image_rgba(source=img2, image='image', x='x', y='y', dw='dw', dh='dh')
points_2 = plot2.circle([100, 90, 180], [100, 90, 180], size=10, color="navy", alpha=0.5)
points2_2 = plot2.circle([150, 96, 180], [150, 60, 380], size=10, color="red", alpha=0.5)
plot2.diamond(x='x', y='y', source=data_points, color='red')

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
img: np.ndarray = cv2.imread('../gato.jpg')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1 = process_image(img1)

img2: np.ndarray = cv2.imread('../bird.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = process_image(img2)

tab1 = Panel(child=p, title="image1")
tab2 = Panel(child=p, title="image2")
t1 = generate_image_tab(img1, img2, 5)
t2 = generate_image_tab(img2, img1, 2)
tabs = [t1, t2]
# layout = Tabs(tabs=tabs)


import sys


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
os.system(command)'''
