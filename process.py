#!/usr/bin/env python3
import pandas as pd
import numpy as np
from rotate import rotate_to_sector_0
import matplotlib.pyplot as plt
import ROOT
import time
import itertools
import random
import mlrose
import sys

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)

infiles = []

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
    
def getModuleHists(HistFile):

    module_hists = []
    inclusive_hists = []
    
    infiles.append(ROOT.TFile.Open(HistFile,"READ"))

    inclusive = {}
    phi60 = {}
    
    for i in range (15): #u
        for j in range (15): #v
            for k in range (53):#layer
                if ( k < 28 and k%2 == 0 ):
                    continue
                inclusive[0,i,j,k] = infiles[-1].Get("ROverZ_silicon_"+str(i)+"_"+str(j)+"_"+str(k) )

    for i in range (15): #u
        for j in range (15): #v
            for k in range (53):#layer
                if ( k < 28 and k%2 == 0 ):
                    continue
                phi60[0,i,j,k] = infiles[-1].Get("ROverZ_Phi60_silicon_"+str(i)+"_"+str(j)+"_"+str(k) )


    for i in range (5): #u
        for j in range (12): #v
            for k in range (37,53):#layer
                inclusive[1,i,j,k] = infiles[-1].Get("ROverZ_scintillator_"+str(i)+"_"+str(j)+"_"+str(k) )

    for i in range (5): #u
        for j in range (12): #v
            for k in range (37,53):#layer
                phi60[1,i,j,k] = infiles[-1].Get("ROverZ_Phi60_scintillator_"+str(i)+"_"+str(j)+"_"+str(k) )


    

    inclusive_hists.append(infiles[-1].Get("ROverZ_Inclusive" ))
    inclusive_hists.append(infiles[-1].Get("ROverZ_Inclusive_Phi60" ))
                
    module_hists.append(inclusive)
    module_hists.append(phi60)
            
    return inclusive_hists,module_hists

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

def getlpGBTHists(data, module_hists):

    lpgbt_hists = []

    for p,phiselection in enumerate(module_hists):#inclusive and phi < 60

        temp = {}

        for lpgbt in range(0,1600) :
            lpgbt_found = False

            lpgbt_hist = ROOT.TH1D( ("lpgbt_ROverZ_silicon_" + str(lpgbt) + "_" + str(p)),"",42,0.076,0.58);
            
            for tpg_index in ['TPGId1','TPGId2']:#lpgbt may be in the first or second position in the file

                for index, row in (data[data[tpg_index]==lpgbt]).iterrows():  
                    lpgbt_found = True
                    
                    if (row['density']==2):#Scintillator

                        hist = phiselection[ 1, row['u'], row['v'], row['layer'] ] # get module hist
                    else:
                        hist = phiselection[ 0, row['u'], row['v'], row['layer'] ] # get module hist        

                    linkfrac = 'TPGeLinkFrac1'
                    if ( tpg_index == 'TPGId2' ):
                        linkfrac = 'TPGeLinkFrac2'

                    lpgbt_hist.Add( hist, row[linkfrac] ) # add module hist with the correct e-link weight

            if lpgbt_found:
                temp[lpgbt] = lpgbt_hist

        lpgbt_hists.append(temp)
    
    return lpgbt_hists

def getMinilpGBTGroups(data):

    minigroups = {}
    counter = 0
    minigroups_swap = {}
    
    for index, row in data.iterrows():
        # if (row['density']==2):#Scintillator
        #     continue
        if (row['nTPG']==0): continue

        elif (row['nTPG']==1): 
            #If there is only one lpgbt attached to the module
            #Check if it is attached to another module
            #If it is not create a new minigroup
            #which is labelled with 'counter'
            if not (row['TPGId1'] in minigroups):

                minigroups[row['TPGId1']] = counter
                counter+=1

            
        elif (row['nTPG']==2): 
            
            #If there are two lpgbts attached to the module
            #The second one is "new"

            if (row['TPGId2'] in minigroups):
                print ("should not be the case?")

            if (row['TPGId1'] in minigroups):
                minigroups[row['TPGId2']] = minigroups[row['TPGId1']]
            else:
                minigroups[row['TPGId1']] = counter
                minigroups[row['TPGId2']] = counter

                counter+=1

    for lpgbt, minigroup in minigroups.items(): 
        if minigroup in minigroups_swap: 
            minigroups_swap[minigroup].append(lpgbt) 
        else: 
            minigroups_swap[minigroup]=[lpgbt] 

    return minigroups,minigroups_swap
    
#def getBundles(minigroups,minigroups_swap,combination):
def getBundles(minigroups_swap,combination):

    #Need to divide the minigroups into 24 groups taking into account their different size
    nBundles = 24
    #The weights are the numbers of lpgbts in each mini-groups
    weights = np.array([ len(minigroups_swap[x])  for x in combination ])
    cumulative_arr = weights.cumsum() / weights.sum()
    #Calculate the indices where to perform the split
    idx = np.searchsorted(cumulative_arr, np.linspace(0, 1, nBundles, endpoint=False)[1:])

    bundles = np.array_split(combination,idx)

    for bundle in bundles:
        weight_bundles = np.array([ len(minigroups_swap[x])  for x in bundle ])
        print ( weight_bundles.sum() )
        if (weight_bundles.sum() > 72 ):
            print ( "Error: more than 72 lpgbts in bundle")

    # for bundle in bundles:
    #     for lpgbt in minigroups_swap[minigroups[bundle[-1]]]:
    #         if not lpgbt in bundle[-5:]:
    #             np.append(bundle,lpgbt)
    #Check the last lpgbt of the bundle, does it belong to a mini-group?    

    
    return bundles
            
def getGroupedlpgbtHists(hists,groups,root=False):

    grouped_lpgbthists = []
    grouped_lpgbthists_list = []

    for p,phiselection in enumerate(hists):

        #temp = {}
        temp_list = {}

        for i in range(len(groups)):#loop over groups

            #Create one lpgbt histogram per big group
            lpgbt_hist = ROOT.TH1D( ("lpgbt_ROverZ_grouped_" + str(i) + "_" + str(p)),"",42,0.076,0.58);
            lpgbt_hist_list = [] 
            
            for lpgbt in groups[i]:#loop over each lpgbt in the big group
                lpgbt_hist.Add( phiselection[lpgbt]  )

            for b in range(1,lpgbt_hist.GetNbinsX()+1): 
                lpgbt_hist_list.append(lpgbt_hist.GetBinContent(b))

            #temp_list[i] = lpgbt_hist
            if (root):
                temp_list[i] = lpgbt_hist
            else:
                temp_list[i] = lpgbt_hist_list


        #grouped_lpgbthists.append(temp)
        grouped_lpgbthists_list.append(temp_list)

    return grouped_lpgbthists_list


def calculateChiSquared(inclusive,grouped):

    chi2_total = 0

    for i in range(2):

        for key,hist in grouped[i].items():

            for b in range(len(hist)):

                squared_diff = np.power(hist[b]-inclusive[i].GetBinContent(b+1)/24, 2 )   

                chi2_total+=squared_diff

    return chi2_total



    

