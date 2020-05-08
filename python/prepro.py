#%%

from __future__ import print_function
from ortools.graph import pywrapgraph
import pandas as pd 
import numpy as np


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

#print(lanes2)

#Getting the transportation costs

tr_cost = pd.read_excel('graph.xlsx', sheet_name='Costs_2' )
tr_cost.columns = tr_cost.iloc[0]
tr_cost.rename(columns = {-1:'Code'}, inplace = True)
tr_cost.drop(columns = ['Code','Direct'], inplace = True)
tr_cost.drop([0,1], inplace = True)
tr_cost.reset_index(drop = True, inplace = True)
tr_cost.sort_values(by=['Plant'] ,inplace = True)
tr_cost.set_index('Plant', inplace = True)
tr_cost2 = tr_cost.T
tr_cost2.sort_index(axis = 0, inplace = True)
tr_cost2.values[[np.arange(tr_cost2.shape[0])]*2] = 0


#building the graph for the OR-tools

start_nodes = []
end_nodes = []
unit_costs  = []

#creating the nodes

for index, row in lanes2.iterrows():
    for i in range(len(row)):
        if row[i] == 1:
            start_nodes.append(index)
            end_nodes.append(row.index[i])
            #print(index,row.index,i)

start_nodes_code = start_nodes.copy()
end_nodes_code = end_nodes.copy()


#replacing the codes of dcs for the numbers

for p in range(len(start_nodes)):
    start_nodes[p] = dcs_df.loc[dcs_df['Dc'] == start_nodes[p],'Id'].iloc[0]
    
for q in range(len(end_nodes)):
    end_nodes[q] = dcs_df.loc[dcs_df['Dc'] == end_nodes[q],'Id'].iloc[0] 

#getting the costs from the cost matrix

for index, row in tr_cost2.iterrows():
    for i in range(len(row)):
        if row[i] > 0:
            unit_costs.append(row[i])


#standarizing costs so OR-tools doesnt die

unit_costs2 = [ i*1000 for i in unit_costs]
unit_costs3 = [int(i) for i in unit_costs2]


capacities = [9999999999]*len(unit_costs)

cols = ['Producing Plant' ,'DC', 'Material', 'Yearly FCS (kg)']
portfolio_out = pd.read_excel('portfolio_16.09.19.xlsx')
portfolio_out = portfolio_out.iloc[0:0]
portfolio = pd.read_excel('portfolio_16.09.19.xlsx', usecols = cols)

#%%
mat = [5082951]
material = portfolio[portfolio['Material'] == mat[0]]
material['Yearly FCS (kg)'] = material['Yearly FCS (kg)'].astype(int)
material.loc[:,'Yearly FCS (kg)'] *= -1
material = material.append(material.sum(numeric_only=True), ignore_index=True)
material.iloc[-1,material.columns.get_loc('Yearly FCS (kg)')] *= -1 
material.iloc[-1,material.columns.get_loc('Producing Plant')] = material.iloc[0,0]
material.iloc[-1,material.columns.get_loc('DC')] = material.iloc[0,0]
material.iloc[-1,material.columns.get_loc('Material')] = material.iloc[0,2]

mat_producing_plants = material['Producing Plant'].unique()
mat_producing_plant = mat_producing_plants[0]

supplies = [0]*42

for index, row in material.iterrows():
    i = dcs_df.loc[dcs_df['Dc'] == row['DC'],'Id'].iloc[0] 
    supplies[i] = row['Yearly FCS (kg)']
    #print( i, row['DC'], row['Yearly FCS (kg)'])

supplies2 = [int(i) for i in supplies]

#%%

#Plotting the network as a chord diagram
#start_nodes_code = start_nodes.copy()
#end_nodes_code = end_nodes.copy()

""" links = pd.DataFrame(list(zip(start_nodes_code, end_nodes_code)), columns = ['source','target'])
links['value'] = 1

hv.Chord(links)

chord = hv.Chord(links)
chord.opts(
    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
               labels='name', node_color=dim('index').str()))
 """
#%%

#here comes the network!

#making everything int

start_nodes = [int(i) for i in start_nodes]
end_nodes = [int(i) for i in end_nodes]
supplies2 = [int(i) for i in supplies2]

# Instantiate a SimpleMinCostFlow solver.
min_cost_flow = pywrapgraph.SimpleMinCostFlow()

  # Add each arc.
for i in range(0, len(start_nodes)):
    min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], unit_costs3[i])

  # Add node supplies.

for i in range(0, len(supplies2)):
    min_cost_flow.SetNodeSupply(i, supplies2[i])


  # Find the minimum cost flow between node 0 and node 4.
if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
    print('Minimum cost:', min_cost_flow.OptimalCost())
    print('')
    print('  Arc    Flow / Capacity  Cost')
    for i in range(min_cost_flow.NumArcs()):
        cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
        if cost > 0:
            d.append({'V1': min_cost_flow.Tail(i),'V2': min_cost_flow.Head(i), 'Flow': min_cost_flow.Flow(i), 'Cost':cost })
          

            print('%1s -> %1s   %3s  / %3s       %3s' % (
            min_cost_flow.Tail(i),
            min_cost_flow.Head(i),
            min_cost_flow.Flow(i),
            min_cost_flow.Capacity(i),
            cost))
    result_network = pd.DataFrame(d)
    result_network_codes = result_network.copy(deep=True)
    result_network['V1'].replace(to_replace=[0,1,2,3,4,11,6,7,8,9,31,5,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,10,34,35,36,37,38,39,40,41], value = ['WPV','F03','N01','GOH','VVD','ZAZ','SMK','GLE','COR','DBL','BNA','RTB','E05','J11','P04','ESS','MMA','P02','BER','P05','PWA','DS2','W21','W23','W22','W11','J04','LPC','CTL','UMB','DNV','MHE','MTA','GRO','DSI','AST','CST','AZO','S02','J01','W12','E03'], inplace=True)
    result_network['V2'].replace(to_replace=[0,1,2,3,4,11,6,7,8,9,31,5,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,10,34,35,36,37,38,39,40,41], value = ['WPV','F03','N01','GOH','VVD','ZAZ','SMK','GLE','COR','DBL','BNA','RTB','E05','J11','P04','ESS','MMA','P02','BER','P05','PWA','DS2','W21','W23','W22','W11','J04','LPC','CTL','UMB','DNV','MHE','MTA','GRO','DSI','AST','CST','AZO','S02','J01','W12','E03'], inplace=True)
    
    portfolio_out['DC'] = result_network['V2']
    NetworkCnt = len(result_network.index)
    portfolio_out['NetworkCnt'] = NetworkCnt
    portfolio_out['Producing Plant'] = mat_producing_plant
    portfolio_out['Supplier'] = result_network['V1']
    portfolio_out['Yearly FCS (kg)'] = result_network['Flow']
    portfolio_out['Material'] = mat
    portfolio_out.fillna('Linearopt', inplace = True)



    print('Finished working on: ', mat)
    result_network.to_excel('result_network.xlsx')
    portfolio_out.to_excel('Linearportfolio.xlsx', index = False)

else:
    print('There was an issue with the min cost flow input.')

#%%
""" 
#Visualizing a specific network in Chord diagram

result_network=result_network[['V1','V2','Flow']]

result_network['value'] = result_network['Flow']
result_network.columns = ['source','target','value']
result_network['value'] = result_network['value'].astype(int)

hv.Chord(result_network)
dcs_df['name'] = dcs_df['Dc']
#%%
#dcs_df['name'] = dcs_df['Dc']
dcs_df['group'] = dcs_df.index
dcs_df.drop(columns = ['Dc','Id'], inplace = True)
#%%
nodes = hv.Dataset(dcs_df,'index')
chord = hv.Chord(result_network)
chord.opts(
    opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
               labels='name', node_color=dim('index').str()))
 """
