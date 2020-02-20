import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)

column_names=['layer', 'u', 'v', 'density', 'nDAQ', 'nTPG','DAQId1','nDAQeLinks1','DAQId2','nDAQeLinks2','TPGId1','nTPGeLinks1','TPGId2','nTPGeLinks2']

#Read in data file
data = pd.read_csv("data/FeMappingV3.txt",delim_whitespace=True,names=column_names)

#Set number of TCs by hand for now (In reality this number is taken per event from CMSSW)
data_tcs_passing = data[['layer', 'u', 'v']].copy();
data_tcs_passing[ 'nTCs' ] = 48

#For a given module you need to know the total number of e-links
data['TPGeLinkSum'] = data[['nTPGeLinks1','nTPGeLinks2']].sum(axis=1).where( data['nTPG'] == 2 , data['nTPGeLinks1'])

#And find the fraction taken by the first or second lpgbt
data['TPGeLinkFrac1'] = np.where( (data['nTPG'] > 0), data['nTPGeLinks1']/data['TPGeLinkSum'], 0  )
data['TPGeLinkFrac2'] = np.where( (data['nTPG'] > 1), data['nTPGeLinks2']/data['TPGeLinkSum'], 0  )

#Loop over all lpgbts

lpgbt_loads=[]

for lpgbt in range(1,1600) :
    
    tc_load = 0.

    for tpg_index in ['TPGId1','TPGId2']:#lpgbt may be in the first or second position in the file
    
        for index, row in (data[data[tpg_index]==lpgbt]).iterrows():  
            if (row['density']==2):#ignore scintillator for now
                continue

            #Get the number of TCs (hardcoded to 48 for now, the number passing threshold is less)
            ntcs = data_tcs_passing[(data_tcs_passing['layer']==row['layer'])&(data_tcs_passing['u']==row['u'])&(data_tcs_passing['v']==row['v'])]['nTCs'].values[0]

            tc_load += row['TPGeLinkFrac1'] * ntcs #Add the number of trigger cells from a given module to the lpgbt

    lpgbt_loads.append(tc_load)
    #print(lpgbt,tc_load)    
        
fig = plt.hist(lpgbt_loads, bins=10)
plt.savefig("hist.png")
