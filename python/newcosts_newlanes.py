#%%

from __future__ import print_function
from ortools.graph import pywrapgraph
import pandas as pd 
import numpy as np

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
tr_cost_old = tr_cost2.copy()
current_dcs = tr_cost2.index.values.tolist()

new_cost = pd.read_csv('Tariffs for python.csv')

newdcs_origin = new_cost['Origin'].unique()
newdcs_destination = new_cost['Destination'].unique()
newdcs = list(newdcs_origin)+( list(newdcs_destination))

for dc in newdcs:
    if dc not in dcs:
        dcs.append(dc)
        lanes2.loc[len(lanes2)] = 0
        tr_cost2.loc[len(tr_cost2)] = 0
        lanesindexnames = lanes2.index.values
        lanesindexnames[-1] = dc
        trcostindexnames = tr_cost2.index.values
        trcostindexnames[-1] = dc
        tr_cost2[dc] = 0
        lanes2[dc] = 0
        

        

        print('Dc not in old dcs: ', dc)

tr_cost2 = tr_cost2.astype(np.float64)
dcs_dct = { i: dcs[i] for i in range(0,len(dcs))}
dcs_df = pd.DataFrame(dcs, columns = ['Dc'])
dcs_df['Id'] = dcs_df.index
current_dcs = tr_cost2.index.values.tolist()

#%%

for index, row in new_cost.iterrows():
    if row['Origin'] in current_dcs and row['Destination'] in current_dcs and row['Origin'] != row['Destination'] :
        #print (row['Origin'], row['Destination'], row['Cost/kg'])
        if tr_cost2.at[row['Origin'],row['Destination']] != row['Cost/kg']:
            if tr_cost2.at[row['Origin'],row['Destination']] == 0 and row['Cost/kg'] > 0 :
                tr_cost2.at[row['Origin'],row['Destination']] = row['Cost/kg']
                
                lanes2.at[row['Origin'],row['Destination']] = 1
                print('New lane created between: ', row['Origin'],' ', row['Destination'],' ', tr_cost2.at[row['Origin'], row['Destination']], ' ', lanes2.at[row['Origin'], row['Destination'] ])
            elif tr_cost2.at[row['Origin'],row['Destination']] != 0 and row['Cost/kg'] == 0 :
                tr_cost2.at[row['Origin'],row['Destination']] = row['Cost/kg']
                lanes2.at[row['Origin'],row['Destination']] = 0
                print('lane deleted!!!!!!!! between: ', row['Origin'],' ', row['Destination'],' ', row['Cost/kg'] )
            elif tr_cost2.at[row['Origin'],row['Destination']] != 0 and row['Cost/kg'] != 0 :
                tr_cost2.at[row['Origin'],row['Destination']] = row['Cost/kg']
                print('lane updated between: ', row['Origin'],' ', row['Destination'] )
    #else:
         #print('Dc not in Original DCs: ', row['Origin'],' ', row['Destination'] )

    
    




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



