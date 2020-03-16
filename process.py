#!/usr/bin/env python3
import pandas as pd
import numpy as np
from rotate import rotate_to_sector_0
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', None)

def loadDataFile(MappingFile):

    column_names=['layer', 'u', 'v', 'density', 'shape', 'nDAQ', 'nTPG','DAQId1','nDAQeLinks1','DAQId2','nDAQeLinks2','TPGId1','nTPGeLinks1','TPGId2','nTPGeLinks2']

    #Read in data file
    data = pd.read_csv(MappingFile,delim_whitespace=True,names=column_names)

    #For a given module you need to know the total number of e-links
    data['TPGeLinkSum'] = data[['nTPGeLinks1','nTPGeLinks2']].sum(axis=1).where( data['nTPG'] == 2 , data['nTPGeLinks1'])

    #And find the fraction taken by the first or second lpgbt
    data['TPGeLinkFrac1'] = np.where( (data['nTPG'] > 0), data['nTPGeLinks1']/data['TPGeLinkSum'], 0  )
    data['TPGeLinkFrac2'] = np.where( (data['nTPG'] > 1), data['nTPGeLinks2']/data['TPGeLinkSum'], 0  )
    return data

def getTCsPassing(CMSSW_Silicon,CMSSW_Scintillator):

    #Set number of TCs by hand for now (In reality this number is taken per event from CMSSW)
    column_names=[ 'u', 'v', 'layer', 'nTCs', 'nWords' ]
    data = pd.read_csv(CMSSW_Silicon,names=column_names)
    data_scin = pd.read_csv(CMSSW_Scintillator,names=column_names)

    return data,data_scin
    
def getlpGBTLoadInfo(data,data_tcs_passing,data_tcs_passing_scin):
    #Loop over all lpgbts
        
    lpgbt_loads_tcs=[]
    lpgbt_loads_words=[]
    layers=[]

    for lpgbt in range(0,1600) :

        tc_load = 0.
        word_load = 0.
        lpgbt_layer = -1.
        lpgbt_found = False
        
        for tpg_index in ['TPGId1','TPGId2']:#lpgbt may be in the first or second position in the file

            for index, row in (data[data[tpg_index]==lpgbt]).iterrows():  
                passing = data_tcs_passing
                if (row['density']==2):#Scintillator
                    passing = data_tcs_passing_scin

                lpgbt_found = True
                lpgbt_layer = row['layer']
                #Get the number of TCs passing threshold
                line = passing[(passing['layer']==row['layer'])&(passing['u']==row['u'])&(passing['v']==row['v'])]
                nwords = -1
                ntcs = -1
                if not line.empty: 
                    nwords = line['nWords'].values[0]
                    ntcs = line['nTCs'].values[0]

                    linkfrac = 'TPGeLinkFrac1'
                    if ( tpg_index == 'TPGId2' ):
                        linkfrac = 'TPGeLinkFrac2'
                
                    tc_load += row[linkfrac] * ntcs #Add the number of trigger cells from a given module to the lpgbt
                    word_load += row[linkfrac] * nwords #Add the number of trigger cells from a given module to the lpgbt
                    
        if (lpgbt_found):
            lpgbt_loads_tcs.append(tc_load)

            lpgbt_loads_words.append(word_load)
            layers.append( lpgbt_layer )

    return lpgbt_loads_tcs,lpgbt_loads_words,layers

def getHexModuleLoadInfo(data,data_tcs_passing,data_tcs_passing_scin,print_modules_no_tcs=False):

    module_loads_words=[]
    layers = []
    u_list = []
    v_list = []

    for index, row in data.iterrows():
        if ( row['TPGeLinkSum'] < 0 ):
            continue
        passing = data_tcs_passing
        if (row['density']==2):
            passing = data_tcs_passing_scin
        
        module_layer = row['layer']
        line = passing[(passing['layer']==row['layer'])&(passing['u']==row['u'])&(passing['v']==row['v'])]

        nwords = -1
        ntcs = -1
        #if not line.empty: 
        nwords = line['nWords'].values[0]
        ntcs = line['nTCs'].values[0]
        if (print_modules_no_tcs):
            if ( ntcs == 0 ):
                if (row['density']<2):
                    print ("sili",row['layer'],row['u'],row['v'])
                if (row['density']==2):
                    print ("scin",row['layer'],row['u'],row['v'])
        
        word_load = nwords / (2 * row['TPGeLinkSum'] )

        module_loads_words.append(word_load)
        layers.append(module_layer)
        u_list.append(row['u'])
        v_list.append(row['v'])

    return module_loads_words,layers,u_list,v_list

def check_for_missing_modules(data,data_tcs_passing,data_tcs_passing_scin):

    mappingfile_sil = data[data['density']<2][['layer', 'u', 'v']]
    mappingfile_scin = data[data['density']==2][['layer', 'u', 'v']]

    cmssw_sil = data_tcs_passing[['u','v','layer','nTCs']]
    cmssw_scin = data_tcs_passing_scin[['u','v','layer','nTCs']]

    #onlymapping_sil = mappingfile.merge(cmssw.drop_duplicates(), on=['u','v','layer'],how='left', indicator=True)
    onlycmssw_sil = cmssw_sil.merge(mappingfile_sil.drop_duplicates(), on=['u','v','layer'],how='left', indicator=True)
    onlycmssw_scin = cmssw_scin.merge(mappingfile_scin.drop_duplicates(), on=['u','v','layer'],how='left', indicator=True)

    onlycmssw_sil = onlycmssw_sil[onlycmssw_sil['_merge'] == 'left_only']
    onlycmssw_scin = onlycmssw_scin[onlycmssw_scin['_merge'] == 'left_only']

    print ("Silicon")
    print (onlycmssw_sil[onlycmssw_sil['nTCs']>0][['layer','u','v']].to_string(index=False))
    print ("Scintillator")
    print (onlycmssw_scin[onlycmssw_scin['nTCs']>0][['layer','u','v']].to_string(index=False))
    
def plot(variable,savename="hist.png",binwidth=1,xtitle='Number of words on a single lpGBT'):
    fig = plt.figure()
    binwidth=binwidth
    plt.hist(variable, bins=np.arange(min(variable), max(variable) + binwidth, binwidth))
    plt.ylabel('Number of Entries')
    plt.xlabel(xtitle)
    plt.savefig(savename)
    

def plot2D(variable_x,variable_y,savename="hist2D.png",binwidthx=1,binwidthy=1,xtitle='Number of words on a single lpGBT'):
    
    fig = plt.figure()
    binwidthx=binwidthx
    binwidthy=binwidthy
    plt.hist2d(variable_x,variable_y,bins=[np.arange(min(variable_x), max(variable_x) + binwidthx, binwidthx),np.arange(min(variable_y), max(variable_y) + binwidthy, binwidthy)])
#    plt.hist2d(variable_x,variable_y,bins=[np.arange(0.9, max(variable_x) + binwidthx, binwidthx),np.arange(min(variable_y), max(variable_y) + binwidthy, binwidthy)])
    plt.colorbar()
    plt.ylabel('Layer')
    plt.xlabel(xtitle)
    plt.savefig(savename)

def main():

    #Customisation
    MappingFile = "data/FeMappingV6.txt"

    #V11
    CMSSW_Silicon = "data/average_tcs_sil_v11_ttbar_20200305.csv"
    CMSSW_Scintillator = "data/average_tcs_scin_v11_ttbar_20200305.csv"

    #V10
    # CMSSW_Silicon = "data/average_tcs_sil_v10_qg_20200305.csv"
    # CMSSW_Scintillator = "data/average_tcs_scin_v10_qg_20200305.csv"

    
    Plot_lpGBTLoads = False
    Plot_ModuleLoads = True
    
    
    #Load Data    
    data = loadDataFile(MappingFile) #dataframe
    data_tcs_passing,data_tcs_passing_scin = getTCsPassing(CMSSW_Silicon,CMSSW_Scintillator) #from CMSSW

    #Check for missing modules
    #check_for_missing_modules(data,data_tcs_passing,data_tcs_passing_scin)

    if ( Plot_lpGBTLoads ):
        lpgbt_loads_tcs,lpgbt_loads_words,lpgbt_layers = getlpGBTLoadInfo(data,data_tcs_passing,data_tcs_passing_scin)
        plot(lpgbt_loads_tcs,"loads_tcs.png",binwidth=0.1,xtitle='Number of TCs on a single lpGBT')
        plot(lpgbt_loads_words,"loads_words.png",binwidth=0.1,xtitle='Number of words on a single lpGBT')
        plot2D(lpgbt_loads_tcs,lpgbt_layers,"tcs_vs_layer.png",xtitle='Number of TCs on a single lpGBT')
        plot2D(lpgbt_loads_words,lpgbt_layers,"words_vs_layer.png",xtitle='Number of words on a single lpGBT')

    if ( Plot_ModuleLoads ):
        module_loads_words,module_layers,u,v = getHexModuleLoadInfo(data,data_tcs_passing,data_tcs_passing_scinprint_modules_no_tcs=False)

        d= {'loads':module_loads_words,'layer':module_layers,'u':u,'v':v}
        df = pd.DataFrame(d)
        result = df.sort_values(['loads'])
        print(result)
    
        plot(module_loads_words,"module_loads_words.png",binwidth=0.01,xtitle=r'Average number of words on a single module / $2 \times N_{e-links}$')
        plot2D(module_loads_words,module_layers,"module_words_vs_layer.png",binwidthx=0.05,binwidthy=1,xtitle=r'Average number of words on a single module / $2 \times N_{e-links}$')
    


    
main()
