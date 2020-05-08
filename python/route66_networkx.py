#%%
from __future__ import print_function
from ortools.graph import pywrapgraph
import pandas as pd 
import numpy as np
import networkx as nx
import plotly.graph_objects as go

import holoviews as hv
from holoviews import opts, dim
from bokeh.sampledata.les_mis import data

hv.extension('bokeh')
hv.output(size=400)

#Initial codings for the dcs

d=[]
bbb = []
dcs = ['WPV','F03','N01','GOH','VVD','RTB','SMK','GLE','COR','DBL','GRO','ZAZ','E05','J11','P04','ESS','MMA','P02','BER','P05','PWA','DS2','W21','W23','W22','W11','J04','LPC','CTL','UMB','DNV','BNA','MHE','MTA','DSI','AST','CST','AZO','S02','J01','W12','E03']
dcs_dct = { i: dcs[i] for i in range(0,len(dcs))}
dcs_df = pd.DataFrame(dcs, columns = ['Dc'])
dcs_df['Id'] = dcs_df.index

#getting the lanes

lanes = pd.read_excel('graph.xlsx', sheet_name='Lanes_2' )
lanes.columns = lanes.iloc[0]
lanes.rename(columns = {-1:'Code'}, inplace = True)
lanes.drop(columns = ['Code','Direct'], inplace = True)
lanes.drop([0,1], inplace = True)
lanes.reset_index(drop = True, inplace = True)
lanes.sort_values(by=['Plant'] ,inplace = True)
lanes.set_index('Plant', inplace = True)
lanes2 = lanes.T
lanes2.sort_index(axis = 0, inplace = True)

lanes2.values[[np.arange(lanes2.shape[0])]*2] = 0
lanes2 = lanes2.apply(pd.to_numeric)
lanes2np = lanes2.to_numpy()
G = nx.from_numpy_matrix(lanes2np)
#G = nx.from_pandas_adjacency(lanes2)
G.name = 'Graph generated from current lines between Dcs'

pos = nx.spring_layout(G, k = 1.2,   iterations = 30)
for n, p in pos.items():
    G.node[n]['pos'] = p



edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.node[edge[0]]['pos']
    x1, y1 = G.node[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.node[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

node_adjacencies = []
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+str(len(adjacencies[1])))

node_trace.marker.color = node_adjacencies
node_trace.text = node_text

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                width=1000,
                height=800,
                title='<br>Route 66 network graphical representation',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Route66",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()

#%%

dc_locs = pd.read_excel('Dc_coordinates.xlsx')