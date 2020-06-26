#!/usr/bin/env python3
import pandas as pd
import numpy as np
from rotate import rotate_to_sector_0
import matplotlib.pyplot as plt
import ROOT
import time
import itertools
import random
import sys
from root_numpy import hist2array

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
    
#Legacy function to read ROverZHistograms file with 1D histograms
def getModuleHists1D(HistFile):

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

def getModuleHists(HistFile):

    module_hists = []
    inclusive_hists = []
    
    infiles.append(ROOT.TFile.Open(HistFile,"READ"))

    phiGreater60 = {}
    phiLess60 = {}
    
    for i in range (15): #u
        for j in range (15): #v
            for k in range (53):#layer
                if ( k < 28 and k%2 == 0 ):
                    continue
                
                PhiVsROverZ = infiles[-1].Get("ROverZ_silicon_"+str(i)+"_"+str(j)+"_"+str(k) )
                #phi<60 is half the total number of bins in the y-dimension, i.e. for 12 bins (default) would be 6
                nBinsPhi = PhiVsROverZ.GetNbinsY()
                phiGreater60[0,i,j,k] = PhiVsROverZ.ProjectionX( "ROverZ_silicon_"+str(i)+"_"+str(j)+"_"+str(k) +"_PhiGreater60", nBinsPhi//2 + 1, nBinsPhi )  
                phiLess60[0,i,j,k] = PhiVsROverZ.ProjectionX( "ROverZ_silicon_"+str(i)+"_"+str(j)+"_"+str(k) +"_PhiLess60", 1, nBinsPhi//2)                


    for i in range (5): #u
        for j in range (12): #v
            for k in range (37,53):#layer
                PhiVsROverZ = infiles[-1].Get("ROverZ_scintillator_"+str(i)+"_"+str(j)+"_"+str(k) )
                #phi<60 is half the total number of bins in the y-dimension, i.e. for 12 bins (default) would be 6
                nBinsPhi = PhiVsROverZ.GetNbinsY()
                phiGreater60[1,i,j,k] =  PhiVsROverZ.ProjectionX( "ROverZ_scintillator_"+str(i)+"_"+str(j)+"_"+str(k) +"_PhiGreater60", nBinsPhi//2 + 1, nBinsPhi )
                phiLess60[1,i,j,k] =  PhiVsROverZ.ProjectionX( "ROverZ_scintillator_"+str(i)+"_"+str(j)+"_"+str(k) +"_PhiLess60", 1, nBinsPhi//2)

    
    PhiVsROverZ = infiles[-1].Get("ROverZ_Inclusive" )
    nBinsPhi = PhiVsROverZ.GetNbinsY()
    inclusive_hists.append(PhiVsROverZ.ProjectionX( "ROverZ_PhiGreater60", nBinsPhi//2 + 1, nBinsPhi ) )
    inclusive_hists.append(PhiVsROverZ.ProjectionX( "ROverZ_PhiLess60" , 1, nBinsPhi//2 ) )
                
    module_hists.append(phiGreater60)
    module_hists.append(phiLess60)
            
    return inclusive_hists,module_hists

def getHistsPerLayer(module_hists):

    f = ROOT.TFile.Open("HistsPerLayer.root","RECREATE")
    histsPerLayer = []
    
    for layer in range(1,53):
        layer_hist = ROOT.TH1D( ("layer_ROverZ" + "_" + str(layer)),"",42,0.076,0.58);
        histsPerLayer.append(layer_hist)

    print (len(histsPerLayer))

    #phi > 60
    for key,hist in module_hists[0].items():
        #key[3] is the layer number (which goes from 1-52)
        histsPerLayer[key[3]-1].Add( hist )

    #phi < 60
    for key,hist in module_hists[1].items():
        #key[3] is the layer number (which goes from 1-52)
        histsPerLayer[key[3]-1].Add( hist )

    for hist in histsPerLayer:
        hist.Write()

    f.Close()
        
    return histsPerLayer

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

def getMiniGroupHists(lpgbt_hists, minigroups_swap,root=False):
    
    minigroup_hists = []

    minigroup_hists_phiGreater60 = {}
    minigroup_hists_phiLess60 = {}

    for minigroup, lpgbts in minigroups_swap.items():
        
        phiGreater60 = ROOT.TH1D( "minigroup_ROverZ_silicon_" + str(minigroup) + "_0","",42,0.076,0.58) 
        phiLess60    = ROOT.TH1D( "minigroup_ROverZ_silicon_" + str(minigroup) + "_1","",42,0.076,0.58) 

        for lpgbt in lpgbts:

            phiGreater60.Add( lpgbt_hists[0][lpgbt] )
            phiLess60.Add( lpgbt_hists[1][lpgbt] )

            
        phiGreater60_array = hist2array(phiGreater60)
        phiLess60_array = hist2array(phiLess60) 

        if ( root ):
            minigroup_hists_phiGreater60[minigroup] = phiGreater60
            minigroup_hists_phiLess60[minigroup] = phiLess60
        else:
            minigroup_hists_phiGreater60[minigroup] = phiGreater60_array
            minigroup_hists_phiLess60[minigroup] = phiGreater60_array
            
    minigroup_hists.append(minigroup_hists_phiGreater60)
    minigroup_hists.append(minigroup_hists_phiLess60)

    return minigroup_hists

def getlpGBTHists(data, module_hists):

    lpgbt_hists = []

    for p,phiselection in enumerate(module_hists):#phi > 60 and phi < 60

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

def find_nearest(array, values):
    indices = []
    for value in values:
        array_subtract = array - value
        index = (np.abs(array_subtract)).argmin()
        if array_subtract[index] > 0: index+=1
        indices.append(index)
        
    return indices

def getBundles(minigroups_swap,combination):

    #Need to divide the minigroups into 24 groups taking into account their different size
    nBundles = 24
    #The weights are the numbers of lpgbts in each mini-groups
    weights = np.array([ len(minigroups_swap[x])  for x in combination ])
    cumulative_arr = weights.cumsum() / weights.sum()
    #Calculate the indices where to perform the split

    #Method 1
    #idx = np.searchsorted(cumulative_arr, np.linspace(0, 1, nBundles, endpoint=False)[1:])
    #Method 2 (improved)
    idx = find_nearest(cumulative_arr, np.linspace(0, 1, nBundles, endpoint=False)[1:])

    bundles = np.array_split(combination,idx)

    for bundle in bundles:
        weight_bundles = np.array([ len(minigroups_swap[x])  for x in bundle ])
        if (weight_bundles.sum() > 72 ):
            print ( "Error: more than 72 lpgbts in bundle")
            
    return bundles
            
def getBundledlpgbtHistsRoot(minigroup_hists,bundles):

    bundled_lpgbthists = []
    bundled_lpgbthists_list = []

    for p,phiselection in enumerate(minigroup_hists):

        temp = {}

        for i in range(len(bundles)):#loop over bundles

            #Create one lpgbt histogram per bundle
            lpgbt_hist = ROOT.TH1D( ("lpgbt_ROverZ_bundled_" + str(i) + "_" + str(p)),"",42,0.076,0.58);
            
            for minigroup in bundles[i]:#loop over each lpgbt in the bundle
                lpgbt_hist.Add( phiselection[minigroup]  )

            temp[i] = lpgbt_hist

        bundled_lpgbthists_list.append(temp)

    return bundled_lpgbthists_list


def getBundledlpgbtHists(minigroup_hists,bundles):

    bundled_lpgbthists = []
    bundled_lpgbthists_list = []

    for p,phiselection in enumerate(minigroup_hists):

        temp_list = {}

        for i in range(len(bundles)):#loop over bundles

            #Create one lpgbt histogram per bundle
            lpgbt_hist_list = np.zeros(42) 
            
            for minigroup in bundles[i]:#loop over each lpgbt in the bundle
                lpgbt_hist_list+= phiselection[minigroup] 

            temp_list[i] = lpgbt_hist_list

        bundled_lpgbthists_list.append(temp_list)

    return bundled_lpgbthists_list



def calculateChiSquared(inclusive,grouped):

    chi2_total = 0
    
    for i in range(len(inclusive)):

        for key,hist in grouped[i].items():

            for b in range(len(hist)):
                
                squared_diff = np.power(hist[b]-inclusive[i].GetBinContent(b+1)/24, 2 )   

                chi2_total+=squared_diff

    return chi2_total
