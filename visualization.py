from bokeh.plotting import figure, output_file, show
from bokeh.models import Panel, Tabs
from bokeh.layouts import row
import cv2
import numpy as np

import numpy

from PIL import Image
from bokeh.plotting import figure, show


width = 500
height = 500


im = Image.open('gato.jpg')
im = im.convert("RGBA")
# uncomment to compare
im.show()
imarray = numpy.array(im)

p = figure(x_range=(0,imarray.shape[0]), y_range=(0,imarray.shape[1]), width=width, height=height)

p.image_rgba(image=[imarray], x=0, y=0, dw=imarray.shape[0], dh=imarray.shape[1])

show(p)

# prepare some data
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]



# output to static HTML file
output_file("image_visualization.html")


'''
#Original image
p1 = figure(title="original image",tools="pan,lasso_select,box_select,wheel_zoom",  x_range=(0,width), y_range=(0,height))
p1.image(image=[img], x=0, y=0, dw=10, dh=10, palette="Spectral11", level="image")
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