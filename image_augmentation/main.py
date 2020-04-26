import jinja2

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.models import Button, CustomJS

plot = figure(plot_width=300, plot_height=300, title="")
plot.toolbar.logo = None
plot.toolbar_location = None

button = Button(label="foo")
button.js_on_click(CustomJS(code="""
      document.getElementById("overlay").style.display = "block";
  """))

curdoc().add_root(row(plot, button))
curdoc().template = jinja2.Template("""
  {% extends base %}

  {% block title %}Overlay Example{% endblock %}

  {% block preamble %}
  <style>
  #overlay {
      position: fixed; /* Sit on top of the page content */
      display: none; /* Hidden by default */
      width: 100%; /* Full width (cover the whole page) */
      height: 100%; /* Full height (cover the whole page) */
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background-color: rgba(0,0,0,0.5); /* Black background with opacity */
      z-index: 2; /* Specify a stack order in case you're using a different order for other elements */
      cursor: pointer; /* Add a pointer on hover */
  }
  </style>
  {% endblock %}
  {% block contents %}
  <div id="overlay"><img src='https://media.giphy.com/media/zZMTVkTeEfeEg/giphy.gif'></div>
  {{ embed(roots[0]) }}
  {% endblock %}
  """)

import os
import sys

command = 'bokeh serve --show ' + sys.argv[0]
os.system(command)

