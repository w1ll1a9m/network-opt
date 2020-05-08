#%%

import pandas as pd 
import numpy as np


import holoviews as hv
from holoviews import opts, dim
from bokeh.sampledata.les_mis import data
#from bokeh.sampledata.airport_routes import routes, airport

hv.extension('bokeh')
hv.output(size=350)

links = pd.read_excel ('route66 network links.xlsx', index = False)
links.drop(columns =['Unnamed: 0'], inplace = True)

links_sample = links.sample(frac = 0.7)
links_sample.value = 0
links.update(links_sample)
links.value = links.value.astype(int)


hv.Chord(links)

chord = hv.Chord(links)
chord.opts(
    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
               labels='index', node_color=dim('index').str()))
