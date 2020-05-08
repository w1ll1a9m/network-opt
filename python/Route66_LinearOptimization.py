from __future__ import print_function
from ortools.graph import pywrapgraph
import pandas as pd 
import numpy as np
from tqdm import tqdm
import pyprind
import psutil

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
tr_cost_old = tr_cost2.copy()

#updating costs and lanes


new_cost = pd.read_excel('tariffs.xlsx')

newdcs_origin = new_cost['Origin'].unique()
newdcs_destination = new_cost['Destination'].unique()
newdcs = list(newdcs_origin)+( list(newdcs_destination))
newnewdcs = []

#adding lanes not present in the adjacency matrix

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

        

        newnewdcs.append(dc)
print('new dcs found: ', newnewdcs)
tr_cost2 = tr_cost2.astype(np.float64)
dcs_dct = { i: dcs[i] for i in range(0,len(dcs))}
dcs_df = pd.DataFrame(dcs, columns = ['Dc'])
dcs_df['Id'] = dcs_df.index
current_dcs = tr_cost2.index.values.tolist()

#updating costs and lanes according to the file

for index, row in new_cost.iterrows():
    if row['Origin'] in current_dcs and row['Destination'] in current_dcs and row['Origin'] != row['Destination'] :
        #print (row['Origin'], row['Destination'], row['Cost/kg'])
        if tr_cost2.at[row['Origin'],row['Destination']] != row['Cost/kg']:
            if tr_cost2.at[row['Origin'],row['Destination']] == 0 and row['Cost/kg'] > 0 :
                tr_cost2.at[row['Origin'],row['Destination']] = row['Cost/kg']
                lanes2.at[row['Origin'],row['Destination']] = 1
                #print('New lane created between: ', row['Origin'],' ', row['Destination'],' ', row['Cost/kg'] )
            elif tr_cost2.at[row['Origin'],row['Destination']] != 0 and row['Cost/kg'] == 0 :
                tr_cost2.at[row['Origin'],row['Destination']] = row['Cost/kg']
                lanes2.at[row['Origin'],row['Destination']] = 0
                #print('lane deleted!!!!!!!! between: ', row['Origin'],' ', row['Destination'],' ', row['Cost/kg'] )
            elif tr_cost2.at[row['Origin'],row['Destination']] != 0 and row['Cost/kg'] != 0 :
                tr_cost2.at[row['Origin'],row['Destination']] = row['Cost/kg']
                #print('lane updated between: ', row['Origin'],' ', row['Destination'] )
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



topmats = 200
cols = ['Producing Plant' ,'DC','NetworkCnt', 'Material', 'Yearly FCS (kg)', 'Supplier Type', 'Supplier']

portfolio = pd.read_excel('Portfolio 11-10-19.xlsx', sheet_name='Top', usecols = cols)
portfolio_head = pd.read_excel('Portfolio 11-10-19.xlsx', sheet_name='Top')
portfolio_head = portfolio_head.iloc[0:0]
finalout = portfolio_head.copy()

portfolio = portfolio[portfolio['Yearly FCS (kg)'] > 1]
#print(portfolio[portfolio['Material']==5031903])
portfolio = portfolio[portfolio['Producing Plant'] != portfolio['DC']]

portfoliogroup = portfolio.groupby('Material').first()
portfoliogroup.sort_values(by = ['NetworkCnt'], ascending = False, inplace = True)
topmaterialsdf = portfoliogroup.head(topmats)
topmaterials = topmaterialsdf.index.values.tolist()
allmaterials = portfoliogroup.index.values.tolist()

#allmaterials.remove(5094168)

#5256756

# for the materials

allmaterials = [5031903]

#tqdm(,, desc ='Processing materials', total = len(allmaterials))

#bar = pyprind.ProgBar(len(allmaterials), monitor=True, bar_char='█')

#print(portfolio[portfolio['Material']==5031903])

for mat in tqdm(allmaterials, desc ='Processing materials', total = len(allmaterials)):
    lastmat = mat
    #bar.update()
    #print(bar)
    d=[]
    portfolio_out = portfolio_head.copy()
    material = portfolio[portfolio['Material'] == mat]
    mat_prodplt = material.loc[material['Supplier Type'] == 'Producing Plant', ['Supplier']]
    mat_producing_plants = mat_prodplt['Supplier'].unique()

    if len(mat_producing_plants) == 0:
        mat_producing_plant = material['Producing Plant'].unique()
        mat_producing_plant = mat_producing_plant[0]
        mat_producing_plants = list(mat_producing_plants)
        mat_producing_plants.append(mat_producing_plant)
    else:
        mat_producing_plants = list(mat_producing_plants)
        mat_producing_plant = mat_producing_plants[0]

    if len(mat_producing_plants) == 1:
        material['Producing Plant'] = mat_producing_plant

    material.drop(columns = ['Supplier', 'Supplier Type'], inplace = True)
    material = material[material['DC'] != material['Producing Plant']]
    material['Yearly FCS (kg)'] = material['Yearly FCS (kg)'].astype(int)
    material.loc[:,'Yearly FCS (kg)'] *= -1

    #new shit
    group = material.groupby(['Producing Plant', 'Material']).sum()
    group.reset_index( inplace = True)
    group['Yearly FCS (kg)'] *= -1 
    group['DC'] = group['Producing Plant']
    material = pd.concat([material, group])
    material.reset_index(inplace = True)

    #mat_producing_plants = material['Producing Plant'].unique()
    #mat_producing_plant = mat_producing_plants[0]

    supplies = [0]*len(current_dcs)

    for index, row in material.iterrows():
        i = dcs_df.loc[dcs_df['Dc'] == row['DC'],'Id'].iloc[0] 
        supplies[i] = row['Yearly FCS (kg)']
        #print( i, row['DC'], row['Yearly FCS (kg)'])

    supplies2 = [int(i) for i in supplies]

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
        #print('Minimum cost:', min_cost_flow.OptimalCost())
        #print('')
        #print('  Arc    Flow / Capacity  Cost')
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

        result_network['V1'].replace(dcs_dct, inplace=True)
        result_network['V2'].replace(dcs_dct, inplace=True)


        portfolio_out['DC'] = result_network['V2']
        NetworkCnt = len(result_network.index)
        portfolio_out['NetworkCnt'] = NetworkCnt
        
        portfolio_out['Supplier'] = result_network['V1']
        portfolio_out['Yearly FCS (kg)'] = result_network['Flow']
        portfolio_out['Material'] = mat
        portfolio_out['Supplier Type'] = 'DC'

        
        portfolio_out['COST'] = result_network['Cost']/1000


        portfolio_out.loc[portfolio_out.Supplier.isin(mat_producing_plants) , 'Supplier Type'] = 'Producing Plant'


        portfolio_out['Producing Plant'] = mat_producing_plant

        portfolio_out.loc[portfolio_out['Supplier Type'] == 'Producing Plant', 'Producing Plant'] = portfolio_out.Supplier


        portfolio_out.fillna('0', inplace = True)



        #print('Finished working on: ', mat)
        
        finalout = finalout.append(portfolio_out, ignore_index = True)

        del result_network, portfolio_out, d

        #result_network.to_excel('result_network.xlsx')
        #portfolio_out.to_excel('Linearportfolio.xlsx', index = False)

    else:
        error = mat
        print('There was an issue with the min cost flow input. material:', mat)
print(material)
print(finalout)

finalout.to_excel('Linearportfolioallmaterials.xlsx', index = False)
