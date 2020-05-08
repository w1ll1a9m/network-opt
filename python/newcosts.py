#%%

from __future__ import print_function
from ortools.graph import pywrapgraph
import pandas as pd 
import numpy as np

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

new_cost = pd.read_excel('Lane parameters.xlsx')



#%%

for index, row in new_cost.iterrows():
    if row['Origin'] in current_dcs and row['Destination'] in current_dcs and row['Origin'] != row['Destination'] :
        #print (row['Origin'], row['Destination'], row['Cost/kg'])
        if tr_cost2.at[row['Origin'],row['Destination']] != row['Cost/kg']:
            if tr_cost2.at[row['Origin'],row['Destination']] == 0 and row['Cost/kg'] > 0 :
                tr_cost2.at[row['Origin'],row['Destination']] = row['Cost/kg']
                print('New lane created between: ', row['Origin'],' ', row['Destination'],' ', row['Cost/kg'] )
            elif tr_cost2.at[row['Origin'],row['Destination']] != 0 and row['Cost/kg'] == 0 :
                tr_cost2.at[row['Origin'],row['Destination']] = row['Cost/kg']
                print('lane deleted!!!!!!!! between: ', row['Origin'],' ', row['Destination'],' ', row['Cost/kg'] )
            elif tr_cost2.at[row['Origin'],row['Destination']] != 0 and row['Cost/kg'] != 0 :
                tr_cost2.at[row['Origin'],row['Destination']] = row['Cost/kg']
                #print('lane updated between: ', row['Origin'],' ', row['Destination'] )
    #else:
         #print('Dc not in Original DCs: ', row['Origin'],' ', row['Destination'] )

    
    







