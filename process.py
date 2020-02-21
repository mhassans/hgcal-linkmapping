#!/usr/bin/env python3
import pandas as pd
import numpy as np
from rotate import rotate
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', None)

def loadDataFile():

    column_names=['layer', 'u', 'v', 'density', 'nDAQ', 'nTPG','DAQId1','nDAQeLinks1','DAQId2','nDAQeLinks2','TPGId1','nTPGeLinks1','TPGId2','nTPGeLinks2']

    #Read in data file
    data = pd.read_csv("data/FeMappingV3.txt",delim_whitespace=True,names=column_names)

    #For a given module you need to know the total number of e-links
    data['TPGeLinkSum'] = data[['nTPGeLinks1','nTPGeLinks2']].sum(axis=1).where( data['nTPG'] == 2 , data['nTPGeLinks1'])

    #And find the fraction taken by the first or second lpgbt
    data['TPGeLinkFrac1'] = np.where( (data['nTPG'] > 0), data['nTPGeLinks1']/data['TPGeLinkSum'], 0  )
    data['TPGeLinkFrac2'] = np.where( (data['nTPG'] > 1), data['nTPGeLinks2']/data['TPGeLinkSum'], 0  )
    return data

def getTCsPassing(data):

    #Set number of TCs by hand for now (In reality this number is taken per event from CMSSW)
    data_tcs_passing = data[['layer', 'u', 'v']].copy()
    data_tcs_passing[ 'nTCs' ] = 48
    return data_tcs_passing
    
def getlpGBTLoadInfo(data,data_tcs_passing):
    #Loop over all lpgbts
        
    lpgbt_loads=[]
    layers=[]

    for lpgbt in range(1,1600) :

        tc_load = 0.

        for tpg_index in ['TPGId1','TPGId2']:#lpgbt may be in the first or second position in the file

            for index, row in (data[data[tpg_index]==lpgbt]).iterrows():  
                if (row['density']==2):#ignore scintillator for now
                    continue

                #Get the number of TCs (hardcoded to 48 for now, the number passing threshold is less)
                rot0 = [row['u'],row['v']]
                rot1 = rotate(row['u'],row['v'],row['layer'],1)
                rot2 = rotate(row['u'],row['v'],row['layer'],2)

#                ntcs = data_tcs_passing[(data_tcs_passing['layer']==row['layer'])&(data_tcs_passing['u']==row['u'])&(data_tcs_passing['v']==row['v'])]['nTCs'].values[0]
                ntcs = data_tcs_passing[(data_tcs_passing['layer']==row['layer'])&((data_tcs_passing['u']==rot0[0])|(data_tcs_passing['u']==rot1[0])|(data_tcs_passing['u']==rot2[0]))&((data_tcs_passing['v']==rot0[1])|(data_tcs_passing['v']==rot1[1])|(data_tcs_passing['v']==rot2[1]))]['nTCs'].values[0]
                tc_load += row['TPGeLinkFrac1'] * ntcs #Add the number of trigger cells from a given module to the lpgbt
        if (tc_load > 1):
            lpgbt_loads.append(tc_load)
            layers.append( row['layer'] )
        #print(lpgbt,tc_load)    
    return lpgbt_loads,layers


def plot(variable,savename="hist.png"):
    fig = plt.figure(0)
    binwidth=1
    plt.hist(variable, bins=np.arange(min(variable), max(variable) + binwidth, binwidth))
    plt.ylabel('Number of Entries')
    plt.xlabel('Number of Trigger Cells on a single lpGBT')
    plt.savefig(savename)
    

def plot2D(variable_x,variable_y,savename="hist2D.png"):
    
    fig = plt.figure(1)
    binwidth=1
    plt.hist2d(variable_x,variable_y,bins=[np.arange(min(variable_x), max(variable_x) + 4*binwidth, 4*binwidth),np.arange(min(variable_y), max(variable_y) + binwidth, binwidth)])
    plt.colorbar()
    plt.ylabel('Layer')
    plt.xlabel('Number of Trigger Cells on a single lpGBT')
    plt.savefig(savename)

def main():

    #Load Data
    
    data = loadDataFile() #dataframe
    data_tcs_passing = getTCsPassing(data) #from CMSSW
    lpgbt_loads,layers = getlpGBTLoadInfo(data,data_tcs_passing)


    #Plot Variables of interest
    plot(lpgbt_loads,"loads.png")
    plot2D(lpgbt_loads,layers,"n_vs_layer.png")


    
main()
