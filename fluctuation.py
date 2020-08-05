#!/usr/bin/env python3
import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm
import math
import pickle
from scipy import optimize
from process import loadDataFile
from process import getPhiSplitIndices
from process import getMinilpGBTGroups,getBundles,getBundledlpgbtHists
from rotate import rotate_to_sector_0
from geometryCorrections import applyGeometryCorrectionsNumpy,loadSiliconNTCCorrectionFile
import time
import yaml
import sys, os

def getMiniModuleGroups(data,minigroups_swap):

    minigroups_modules = {}
    
    for minigroup_id,lpgbts in minigroups_swap.items():

        module_list = []

        for lpgbt in lpgbts:

            data_list = data[ ((data['TPGId1']==lpgbt) | (data['TPGId2']==lpgbt)) ]

            for index, row in data_list.iterrows():

                if ( row['density']==2 ):
                    mod = [1, row['u'], row['v'], row['layer']]
                else:
                    mod = [0, row['u'], row['v'], row['layer']]

                if mod not in module_list:
                    module_list.append(mod)

        minigroups_modules[minigroup_id] = module_list
        
    return minigroups_modules
    

def getMiniGroupHistsNumpy(module_hists, minigroups_modules):
    
    minigroup_hists = []

    minigroup_hists_inclusive = {}
    minigroup_hists_phi60 = {}

    for minigroup, modules in minigroups_modules.items():
        
        inclusive = np.zeros(42)
        phi60 = np.zeros(42)

        for module in modules:
            inclusive = inclusive + module_hists[0][module[0],module[1],module[2],module[3]]
            phi60 = phi60 + module_hists[1][module[0],module[1],module[2],module[3]]

        minigroup_hists_inclusive[minigroup] = inclusive.copy()
        minigroup_hists_phi60[minigroup] = phi60.copy()
            
    minigroup_hists.append(minigroup_hists_inclusive)
    minigroup_hists.append(minigroup_hists_phi60)

    return minigroup_hists

def getROverZPhi(x, y, z, sector = 0):

    if (z > 0):
        x = x*-1
    
    r = math.sqrt( x*x + y*y  )
    phi = np.arctan2(y,x)
    
    if (sector == 1):
        if ( phi < np.pi and phi > 0):
            phi = phi-(2*np.pi/3)
        else:
            phi = phi+(4*np.pi/3)
    elif (sector == 2):
        phi = phi+(2*np.pi/3)

    roverz_phi = [r/z,phi]
    return roverz_phi

def etaphiMapping(layer, etaphi):

    if (etaphi[1] > 24 and etaphi[1] <= 72):
        sector = 0
    elif (etaphi[1] > 72 and etaphi[1] <= 120):
        sector = 2
    else:
        sector = 1
        
    if (sector==0):
        pp=etaphi[1]-24
    elif (sector==2):
        pp=etaphi[1]-72
    elif (sector==1):
        if (etaphi[1]<=24):
            etaphi[1] = etaphi[1]+144
        pp = etaphi[1]-120
  
    pp = (pp-1)//4# //Phi index 1-12
  
    if ( etaphi[0] <= 3 ):
        ep = 0
    elif ( etaphi[0] <= 9 ):
        ep = 1
    elif ( etaphi[0] <= 13 ):
        ep = 2
    elif ( etaphi[0] <= 17 ):
        ep = 3
    else:
        ep = 4

    return [ep,pp],sector




def checkFluctuations(initial_state, cmsswNtuple, mappingFile, outputName="alldata", correctionConfig=None, phisplitConfig=None, beginEvent = -1, endEvent = -1):

    nROverZBins = 42
    #To get binning for r/z histograms
    inclusive_hists = np.histogram( np.empty(0), bins = nROverZBins, range = (0.076,0.58) )
    roverzBinning = (inclusive_hists[1])[:-1]
    
    #List of which minigroups are assigned to each bundle 
    init_state = np.hstack(np.load(initial_state,allow_pickle=True))

    #Load the CMSSW ntuple to get per event and per trigger cell information
    rootfile = ROOT.TFile.Open( cmsswNtuple , "READ" )
    tree = rootfile.Get("HGCalTriggerNtuple")

    #Load mapping file
    data = loadDataFile(mappingFile) 

    #Load geometry corrections
    if correctionConfig['nTCCorrectionFile'] != None:
        modulesToCorrect = loadSiliconNTCCorrectionFile( correctionConfig['nTCCorrectionFile'] )
    else:
        modulesToCorrect = pd.DataFrame()
        
    #Get list of which lpgbts are in each minigroup
    minigroups,minigroups_swap = getMinilpGBTGroups(data)

    #Get list of which modules are in each minigroup
    minigroups_modules = getMiniModuleGroups(data,minigroups_swap)
    bundles = getBundles(minigroups_swap,init_state)

    bundled_lpgbthists_allevents = []
    
    ROverZ_per_module_RegionA = {} #traditionally phi > 60 degrees
    ROverZ_per_module_RegionB = {} #traditionally phi < 60 degrees

    #Value of split in phi (nominally 60 degrees)
    if phisplitConfig == None:
        phi_split_RegionA = np.full( nROverZBins, np.pi/3 )
        phi_split_RegionB = np.full( nROverZBins, np.pi/3 )
    else:
        if phisplitConfig['type'] == "fixed":
            phi_split_RegionA = np.full( nROverZBins, np.radians(phisplitConfig['RegionA_fixvalue_min']) )
            phi_split_RegionB = np.full( nROverZBins, np.radians(phisplitConfig['RegionB_fixvalue_max']) )
        else:
            file_roverz_inclusive = ROOT.TFile(str(phisplitConfig['splitfile']),"READ")
            PhiVsROverZ_Total = file_roverz_inclusive.Get("ROverZ_Inclusive" )
            split_indices_RegionA = getPhiSplitIndices( PhiVsROverZ_Total, split = "per_roverz_bin")
            split_indices_RegionB = getPhiSplitIndices( PhiVsROverZ_Total, split = "per_roverz_bin")
            phi_split_RegionA = np.zeros( nROverZBins )
            phi_split_RegionB = np.zeros( nROverZBins )
            for i,(idxA,idxB) in enumerate(zip(split_indices_RegionA, split_indices_RegionB)):
                phi_split_RegionA[i] = PhiVsROverZ_Total.GetYaxis().GetBinLowEdge(int(idxA))
                phi_split_RegionB[i] = PhiVsROverZ_Total.GetYaxis().GetBinLowEdge(int(idxB))
                
    for z in (-1,1):
        for sector in (0,1,2):
            key1 = (z,sector)
            ROverZ_per_module_RegionA[key1] = {}
            ROverZ_per_module_RegionB[key1] = {}

            for i in range (15):
                for j in range (15):
                    for k in range (1,53):
                        if  k < 28 and k%2 == 0:
                            continue
                        ROverZ_per_module_RegionA[key1][0,i,j,k] = np.empty(0)
                        ROverZ_per_module_RegionB[key1][0,i,j,k] = np.empty(0)

            for i in range (5):
                for j in range (12):
                    for k in range (37,53):
                        ROverZ_per_module_RegionA[key1][1,i,j,k] = np.empty(0)
                        ROverZ_per_module_RegionB[key1][1,i,j,k] = np.empty(0)
    
    try:
        for entry,event in enumerate(tree):

            if ( beginEvent != -1 and entry < beginEvent ):
                if ( entry == 0 ):
                    print ("Event number less than " + str(beginEvent) + ", continue" )
                continue;

            if ( endEvent != -1 and entry > endEvent ):
                print ("Event number greater than " + str(endEvent) + ", break" )
                break;

            # if entry > 10:
            #     break
            print ("Event number " + str(entry))


            for key1 in ROverZ_per_module_RegionA.keys():
                for key2 in ROverZ_per_module_RegionA[key1].keys():
                    ROverZ_per_module_RegionA[key1][key2] = np.empty(0)
                    ROverZ_per_module_RegionB[key1][key2] = np.empty(0)
                        
            #Loop over list of trigger cells in a particular
            #event and fill R/Z histograms for each module
            #For RegionA and RegionB (traditionally phi > 60 degrees and phi < 60 degrees respectively)

            for u,v,layer,x,y,z,cellu,cellv in zip(event.tc_waferu,event.tc_waferv,event.tc_layer,event.tc_x,event.tc_y,event.tc_z,event.tc_cellu,event.tc_cellv):

                if ( u > -990 ): #Silicon
                    uv,sector = rotate_to_sector_0(u,v,layer)
                    roverz_phi = getROverZPhi(x,y,z,sector)
                    roverz_bin = np.argmax( roverzBinning > abs(roverz_phi[0]) )

                    if (roverz_phi[1] >= phi_split_RegionA[roverz_bin-1]):
                        #There should be no r/z values lower than 0.076
                        ROverZ_per_module_RegionA[np.sign(z),sector][0,uv[0],uv[1],layer] = np.append(ROverZ_per_module_RegionA[np.sign(z),sector][0,uv[0],uv[1],layer],abs(roverz_phi[0]))            
                    elif (roverz_phi[1] < phi_split_RegionB[roverz_bin-1]):
                        ROverZ_per_module_RegionB[np.sign(z),sector][0,uv[0],uv[1],layer] = np.append(ROverZ_per_module_RegionB[np.sign(z),sector][0,uv[0],uv[1],layer],abs(roverz_phi[0]))
                        
                else: #Scintillator  
                    eta = cellu
                    phi = cellv
                    etaphi,sector = etaphiMapping(layer,[eta,phi])
                    roverz_phi = getROverZPhi(x,y,z,sector)
                    roverz_bin = np.argmax( roverzBinning > abs(roverz_phi[0]) )
                    
                    if (roverz_phi[1] >= phi_split_RegionA[roverz_bin-1]):
                        ROverZ_per_module_RegionA[np.sign(z),sector][1,etaphi[0],etaphi[1],layer] = np.append(ROverZ_per_module_RegionA[np.sign(z),sector][1,etaphi[0],etaphi[1],layer],abs(roverz_phi[0]))            
                    elif (roverz_phi[1] < phi_split_RegionB[roverz_bin-1]):
                        ROverZ_per_module_RegionB[np.sign(z),sector][1,etaphi[0],etaphi[1],layer] = np.append(ROverZ_per_module_RegionB[np.sign(z),sector][1,etaphi[0],etaphi[1],layer],abs(roverz_phi[0]))


            #Bin the TC module data
            module_hists_phigreater60 = {}
            module_hists_philess60 = {}

            for key1,value1 in ROverZ_per_module_RegionA.items():
                module_hists_phigreater60[key1] = {}
                for key2,value2 in value1.items():
                    module_hists_phigreater60[key1][key2] = np.histogram( value2, bins = nROverZBins, range = (0.076,0.58) )[0]
                    
            for key1,value1 in ROverZ_per_module_RegionB.items():
                module_hists_philess60[key1] = {}
                for key2,value2 in value1.items():
                    module_hists_philess60[key1][key2] = np.histogram( value2, bins = nROverZBins, range = (0.076,0.58) )[0]


            for z in (-1,1):
                for sector in (0,1,2):
                        
                    #the module hists are a numpy array of size 42
                    module_hists = [module_hists_phigreater60[z,sector],module_hists_philess60[z,sector]]

                    #Apply geometry corrections
                    applyGeometryCorrectionsNumpy( module_hists, modulesToCorrect )

                    #Sum the individual module histograms to get the minigroup histograms
                    minigroup_hists = getMiniGroupHistsNumpy(module_hists,minigroups_modules)

                    #Sum the minigroup histograms to get the bundle histograms
                    bundled_lpgbthists = getBundledlpgbtHists(minigroup_hists,bundles)

                    bundled_lpgbthists_allevents.append(bundled_lpgbthists)

    except KeyboardInterrupt:
        print("interrupt received, stopping and saving")

    finally:

        #Write all data to file for later analysis (Pickling)
        if ( beginEvent != -1 ):
            outputName = outputName + "_from" + str(beginEvent)
        if ( endEvent != -1 ):
            outputName = outputName + "_to" + str(endEvent)
        
        with open( outputName + ".txt", "wb") as filep:
            pickle.dump(bundled_lpgbthists_allevents, filep)



def plotMeanMax(eventData, outdir = ".", includePhi60 = True):
    #Load pickled per-event bundle histograms
    with open(eventData, "rb") as filep:   
        bundled_lpgbthists_allevents = pickle.load(filep)
    os.system("mkdir -p " + outdir)
    
    nbins = 42
    #To get binning for r/z histograms
    inclusive_hists = np.histogram( np.empty(0), bins = nbins, range = (0.076,0.58) )

    #Names for inclusive and phi < 60 indices
    inclusive = 0
    phi60 = 1

    hists_max = [] 
    
    #Plotting Max, mean and standard deviation per bundle:

    for bundle in range(24):

        list_over_events_inclusive = np.empty(((len(bundled_lpgbthists_allevents)),nbins))
        list_over_events_phi60 = np.empty(((len(bundled_lpgbthists_allevents)),nbins))
        
        for e,event in enumerate(bundled_lpgbthists_allevents):
            list_over_events_inclusive[e] = np.array(event[inclusive][bundle])/6
            list_over_events_phi60[e] = np.array(event[phi60][bundle])/6

        list_over_events_maximum = np.maximum(list_over_events_inclusive, list_over_events_phi60*2 )

        if ( includePhi60 ):
            list_over_events = list_over_events_maximum
        else:
            list_over_events = list_over_events_inclusive

        hist_max = np.amax(list_over_events,axis=0)
        hist_mean = np.mean(list_over_events, axis=0)
        hist_std = np.std(list_over_events, axis=0)

        for s,std in enumerate(hist_std):
            hist_std[s] = std + hist_mean[s]

        pl.bar((inclusive_hists[1])[:-1], hist_max, width=0.012,align='edge')
        pl.bar((inclusive_hists[1])[:-1], hist_std, width=0.012,align='edge')
        pl.bar((inclusive_hists[1])[:-1], hist_mean, width=0.012,align='edge')

        #Plot all events for a given bundle on the same plot
        # for e,event in enumerate(list_over_events):
        #     pl.bar((inclusive_hists[1])[:-1], event, width=0.012,fill=False)
        #     #if (e>200): break

        pl.ylim((0,31))
        pl.savefig( outdir + "/bundle_" + str(bundle) + "max.png" )        
        pl.clf()

        hists_max.append(hist_max)
        
    #Plot maxima for all bundles on the same plot
    for hist in hists_max:
        pl.bar((inclusive_hists[1])[:-1], hist, width=0.012,align='edge')
    pl.ylim((0,31))
    pl.xlabel('r/z')
    pl.ylabel('Maximum number of TCs per bin')
    pl.savefig( outdir + "/maxima.png" )
    pl.clf()

def sumMaximumOverAllEventsAndBundles(truncation,data):
    #Solve for the truncation factor, given two datasets, A and B (data[0] and data[1])
    #And the maximum number of trigger cells allowed in each dataset (data[2] and data[3] respectively)
    #The truncation parameter must be less than or equal to 1
    maximum_A = np.amax(data[0],axis=(1,0))
    maximum_B = np.amax(data[1],axis=(1,0))

    Bscaling_factor = data[2]/data[3]
    
    maxAB = np.maximum(maximum_A,maximum_B*Bscaling_factor)
    
    #Use ceiling rather than round to get worst case
    overallsum_A = np.sum(np.amax(np.where(data[0]<truncation*maxAB,data[0],np.where(truncation*maxAB<maximum_A,truncation*maxAB,maximum_A)),axis=(1,0)))
    overallsum_B = np.sum(np.amax(np.where(data[1]<truncation*(maxAB/Bscaling_factor),data[1],np.where(truncation*(maxAB/Bscaling_factor)<maximum_B,truncation*maxAB/Bscaling_factor,maximum_B)),axis=(1,0)))

    valA = data[2] - overallsum_A
    valB = data[3] - overallsum_B

    #Give a preference that the sum is less than the maximum allowed
    if ( valA < 0 ):
         valA = valA * -1.5
    if ( valB < 0 ):
         valB = valB * -1.5

    optval = valA + valB
    return optval

def plot_NTCs_Vs_ROverZ(inputdata,axis,savename,truncation_curves=None,scaling=None):

    #Fill a 2D histogram per bunch-crossing with N_TCs (maximum over bundles) 

    #Each row represents the r/z bins in a bundle, there are n_bundles*n_events rows
    data = inputdata.reshape(-1,inputdata.shape[-1])

    #Swap axes, such that each row represents an r/z bin, there are n_roverz_bins rows (later flattened)
    data_swap = np.swapaxes(data,0,1)

    #Get the r/z bin axis indices, n_bundles*n_events*[0]+n_bundles*n_events*[1]+...n_bundles*n_events*[n_roverz_bins]
    axis_indices = np.where(data_swap==data_swap)[0]
    #Then get the roverz bin values corresponding to the indices
    roverz = np.take(axis,axis_indices)

    #Plot the 2D histogram
    pl.clf()
    pl.hist2d( roverz , data_swap.flatten() , bins = (len(axis)-1,50),range=[[0.076,0.58], [0, 50]],norm=LogNorm())
    pl.colorbar().set_label("Number of Events")
    #Plot the various 1D truncation curves
    colours = ['red','orange','cyan','green','teal','darkviolet']

    if ( truncation_curves is not None ):
        for t,truncation_option in enumerate(truncation_curves):
            scale = 1.
            if (scaling is not None):
                scale=scaling[t]
            pl.step(axis,np.append(truncation_option,truncation_option[-1])/scale , where = 'post' , color=colours[t],linewidth='3')
            
    pl.xlabel('r/z')
    pl.ylabel('Number of TCs')
    pl.savefig( savename + ".png" )
    pl.clf()

def plot_frac_Vs_ROverZ( dataA, dataB, truncation_curve, TCratio, axis, savename ):

    #Sum over all events and bundles of TCs (in each R/Z bin) 
    totalsumA = np.sum( dataA , axis=(0,1) )
    totalsumB = np.sum( dataB , axis=(0,1) )

    #Sum over all events and bundles of truncated TCs (in each R/Z bin) 
    truncatedsum_A = np.sum(np.where(dataA<truncation_curve, dataA, truncation_curve),axis=(0,1))
    truncatedsum_B = np.sum(np.where(dataB<truncation_curve/TCratio, dataB, truncation_curve/TCratio),axis=(0,1))

    # print ( truncatedsum_A )
    # print ( np.sum((np.where(dataA<truncation_curve, dataA, truncation_curve)),axis=(0,1)) )

    #truncatedsum_B = np.sum(np.floor(np.where(dataB<truncation_curve/TCratio, dataB, truncation_curve/TCratio)),axis=(0,1))

    #Divide to get the fraction, taking into account division by zero
    ratioA = np.divide(   truncatedsum_A, totalsumA , out=np.ones_like(truncatedsum_A), where=totalsumA!=0 )
    ratioB = np.divide(   truncatedsum_B, totalsumB , out=np.ones_like(truncatedsum_B), where=totalsumB!=0 )

    pl.clf()
    pl.step(axis,np.append( ratioA , ratioA[-1] ),color='red',linewidth='1', where = 'post', label='data A')
    pl.step(axis,np.append( ratioB , ratioB[-1] ),color='orange',linewidth='1', where = 'post', label='data B')
    pl.xlabel('r/z')
    pl.ylabel('Sum truncated TCs / Sum all TCs')
    pl.ylim((0.8,1.05))
    pl.legend()
    pl.savefig( savename + ".png" )
    pl.clf()

    
    
def getTruncationValuesRoverZ(data_A, data_B, maxtcs_A, maxtcs_B):
    #Get an array of size nROverZbins, which indicates the maximum number of TCs allowed in each RoverZ bin (the values for data_B are exactly half) (still to consider odd values)
    
    #'scalar' is the value by which the maximum (of data_A or data_B x TCratio) is multiplied to get the truncation values
    result = optimize.minimize_scalar(sumMaximumOverAllEventsAndBundles,args=[data_A, data_B, maxtcs_A, maxtcs_B],bounds=(0.01,2.0),method='bounded')
    scalar = result.x

    maximum_A = np.amax(data_A,axis=(1,0))
    maximum_B = np.amax(data_B,axis=(1,0))


    #Find the floating-point truncation values and round down to make these integers.
    #This will be less than the allowed total due to rounding down.
    truncation_float = np.maximum(maximum_A,maximum_B*(maxtcs_A/maxtcs_B)) * scalar
    truncation_floor = np.floor(truncation_float)
    #The integer difference from the allowed total gives the number of bins that should have their limit incremented by 1.
    integer_difference = np.floor(np.sum(truncation_float)-np.sum(truncation_floor))
    #Find the N bins which have the biggest difference between the floating-point truncation values and the rounded integer
    #and add 1 to these. This gives limits for A, and for B (divided by TC ratio)
    arg = np.argsort(truncation_floor-truncation_float)[:int(integer_difference)]
    truncation_floor[arg]+=1

    return truncation_floor

def loadFluctuationData(eventData):
    #Load the per-event flucation data produced using 'checkFluctuations'
    #Return two arrays (for regions A and B) containing for each event and
    #bundle, the number of TCs in each R/Z bin
    
    with open(eventData, "rb") as filep:   
        bundled_lpgbthists_allevents = pickle.load(filep)
    
    #Names for phi > 60 and phi < 60 indices
    dataA = 0
    dataB = 1

    nbundles = len(bundled_lpgbthists_allevents[0][0]) #24
    nbins = len(bundled_lpgbthists_allevents[0][0][0]) #42
    
    dataA_bundled_lpgbthists_allevents = np.empty((len(bundled_lpgbthists_allevents),nbundles,nbins))
    dataB_bundled_lpgbthists_allevents = np.empty((len(bundled_lpgbthists_allevents),nbundles,nbins))

    for e,event in enumerate(bundled_lpgbthists_allevents):        
        dataA_bundled_lpgbthists_allevents[e] = np.array(list(event[dataA].values()))
        dataB_bundled_lpgbthists_allevents[e] = np.array(list(event[dataB].values()))

    return dataA_bundled_lpgbthists_allevents,dataB_bundled_lpgbthists_allevents
    
def studyTruncationOptions(eventData, outdir = ".", includePhi60 = True):
    
    #Load pickled per-event bundle histograms
    phigreater60_bundled_lpgbthists_allevents,philess60_bundled_lpgbthists_allevents = loadFluctuationData(eventData)

    os.system("mkdir -p " + outdir)

    nbinsROverZ = len(phigreater60_bundled_lpgbthists_allevents[0][0]) #42
    
    #To get binning for r/z histograms
    inclusive_hists = np.histogram( np.empty(0), bins = nbinsROverZ, range = (0.076,0.58) )

    #taking for each bin the maximum of the inclusive and phi60 x 2
    inclusive_bundled_lpgbthists_allevents = phigreater60_bundled_lpgbthists_allevents + philess60_bundled_lpgbthists_allevents

    print ("Get truncation value for option 1")
    truncation_option_1 = getTruncationValuesRoverZ(inclusive_bundled_lpgbthists_allevents,philess60_bundled_lpgbthists_allevents,400,200)
    print ("Get truncation value for option 2")
    truncation_option_2 = getTruncationValuesRoverZ(inclusive_bundled_lpgbthists_allevents,philess60_bundled_lpgbthists_allevents,400,302)
    print ("Get truncation value for option 3")
    truncation_option_3 = getTruncationValuesRoverZ(inclusive_bundled_lpgbthists_allevents,philess60_bundled_lpgbthists_allevents,714,357)

    #Once we have the truncation values, need to find how many TCs are lost
    print ("Plotting histograms")
    #Fill a 2D histogram per bunch-crossing with N_TCs (maximum over bundles) 
    plot_NTCs_Vs_ROverZ(inclusive_bundled_lpgbthists_allevents,inclusive_hists[1],"NTCs_Vs_ROverZ_A",[truncation_option_1,truncation_option_2,truncation_option_3])
    plot_NTCs_Vs_ROverZ(philess60_bundled_lpgbthists_allevents,inclusive_hists[1],"NTCs_Vs_ROverZ_B",[truncation_option_1,truncation_option_2,truncation_option_3],[2,400/302,2])

    #Plot sum of truncated TCs over the sum of all TCs
    plot_frac_Vs_ROverZ( inclusive_bundled_lpgbthists_allevents, philess60_bundled_lpgbthists_allevents, truncation_option_1, 2, inclusive_hists[1], "frac_option_1")
    plot_frac_Vs_ROverZ( inclusive_bundled_lpgbthists_allevents, philess60_bundled_lpgbthists_allevents, truncation_option_2, 400/302, inclusive_hists[1], "frac_option_2")
    plot_frac_Vs_ROverZ( inclusive_bundled_lpgbthists_allevents, philess60_bundled_lpgbthists_allevents, truncation_option_3, 2, inclusive_hists[1], "frac_option_3")
    


def plotTruncation(eventData, outdir = ".", includePhi60 = True):
    #Load pickled per-event bundle histograms
    phigreater60_bundled_lpgbthists_allevents,philess60_bundled_lpgbthists_allevents = loadFluctuationData(eventData)

    #To get binning for r/z histograms
    inclusive_hists = np.histogram( np.empty(0), bins = 42, range = (0.076,0.58) )

    #Form the intersection of the inclusive and phi60 arrays,
    #taking for each bin the maximum of the inclusive and phi60 x 2
    inclusive_bundled_lpgbthists_allevents = phigreater60_bundled_lpgbthists_allevents + philess60_bundled_lpgbthists_allevents
    maximum_bundled_lpgbthists_allevents = np.maximum(inclusive_bundled_lpgbthists_allevents,philess60_bundled_lpgbthists_allevents*2)
    #maximum_bundled_lpgbthists_allevents = np.maximum(inclusive_bundled_lpgbthists_allevents,phigreater60_bundled_lpgbthists_allevents*2)
    
    if ( includePhi60 ):
        hists_max = np.amax(maximum_bundled_lpgbthists_allevents,axis=1)
    else:
        hists_max = np.amax(inclusive_bundled_lpgbthists_allevents,axis=1)
            
    #Find the maximum per bin over all events,
    #Then find the 99th percentile for a 1% truncation

    overall_max = np.amax(hists_max, axis=0)    
    
    overall_max99p = np.round(np.percentile(hists_max,99,axis=0))
    overall_max95p = np.round(np.percentile(hists_max,95,axis=0))
    overall_max90p = np.round(np.percentile(hists_max,90,axis=0))

    #Loop back over events, counting the maximum wait time
    #for each bin, with and without truncation
    total_per_event = []
    total_per_event99 = []
    total_per_event95 = []
    total_per_event90 = []

    max_per_event_perbin = []
    max_per_event_perbin99 = []
    max_per_event_perbin90 = []
    max_per_event_perbin95 = []
    
    for bundle_hists_phigreater60,bundle_hists_philess60 in zip(phigreater60_bundled_lpgbthists_allevents,philess60_bundled_lpgbthists_allevents):

        bundle_hists_inclusive = bundle_hists_philess60 + bundle_hists_phigreater60
        bundle_hists_maximum = np.maximum(bundle_hists_inclusive,bundle_hists_philess60*2)
        #bundle_hists_maximum = np.maximum(bundle_hists_inclusive,bundle_hists_phigreater60*2)
        #24 arrays, with length of 42
        
        sum99 = []
        sum95 = []
        sum90 = []

        if ( includePhi60 ):
            bundle_hists = bundle_hists_maximum
        else:
            bundle_hists = bundle_hists_inclusive

        for bundle in bundle_hists:
            
            #If a given r/z bin is greater than the maximum allowed by truncation then set to the truncated value
            sum99.append ( np.where( np.less( overall_max99p, bundle ), overall_max99p, bundle )  )
            sum95.append ( np.where( np.less( overall_max95p, bundle ), overall_max95p, bundle )  )
            sum90.append ( np.where( np.less( overall_max90p, bundle ), overall_max90p, bundle )  )
        

        total_per_event.append( np.sum(bundle_hists, axis=1 ))        #array with length of 24 (sum over the 42 bins)
        total_per_event99.append( np.sum(sum99, axis=1 ))
        total_per_event95.append( np.sum(sum95, axis=1 ))
        total_per_event90.append( np.sum(sum90, axis=1 ))

        max_per_event_perbin.append( np.amax(bundle_hists, axis=0 ) )
        max_per_event_perbin99.append( np.amax(sum99, axis=0 ) )
        max_per_event_perbin95.append( np.amax(sum95, axis=0 ) )
        max_per_event_perbin90.append( np.amax(sum90, axis=0 ) )

    #Calculating the best possible given the per-event fluctuations

    #For a given r/z bin calculate the mean over all events
    #and add 2.5x the average of the 24 bundles' RMS
    best_likely = np.mean(inclusive_bundled_lpgbthists_allevents,axis=(0,1))+2.5*np.mean(np.std(inclusive_bundled_lpgbthists_allevents,axis=(0)),axis=0)
    ratio_to_best = np.divide(overall_max99p,best_likely,out=np.zeros_like(overall_max99p),where=best_likely!=0)
    
    print ("Maximum TC in any bundle in any event (per bin) = ", np.round(np.amax(max_per_event_perbin,axis=0)))
    print ("Sum of per-bin maximum TC (over bundles and events) = ",  np.round(np.sum(np.amax(max_per_event_perbin,axis=0))))
    print ("Sum of per-bin maximum TC (over bundles and events) with 1% truncation =", np.round(np.sum(np.amax(max_per_event_perbin99,axis=0))))
    print ("Sum of per-bin maximum TC (over bundles and events) with 5% truncation = ", np.round(np.sum(np.amax(max_per_event_perbin95,axis=0))))
    print ("Sum of per-bin maximum TC (over bundles and events) with 10% truncation = ", np.round(np.sum(np.amax(max_per_event_perbin90,axis=0))))

    pl.hist(np.sum(np.array(total_per_event)-np.array(total_per_event99),axis=1)/(24),50,(0,5),histtype='step',log=True,label='1% truncation')
    pl.hist(np.sum(np.array(total_per_event)-np.array(total_per_event95),axis=1)/(24),50,(0,5),histtype='step',log=True,label='5% truncation')
    pl.hist(np.sum(np.array(total_per_event)-np.array(total_per_event90),axis=1)/(24),50,(0,5),histtype='step',log=True,label='10% truncation')    
    pl.xlabel('Number of TCs truncated on average per bundle')
    
    pl.ylabel('Number of Events')
    pl.legend()
    pl.savefig( outdir + "/truncation.png" )

    pl.clf()
    pl.step((inclusive_hists[1])[:-1], ratio_to_best, where='post')
    pl.axhline(y=1, color='r', linestyle='--')
    pl.xlabel('r/z')
    pl.ylabel('Ratio of 1% truncation to likely best')
    pl.ylim((0,10))
    pl.savefig( outdir + "/ratio_to_best.png" )

    #As a cross check plot the bundle R/Z histograms integrated over all events.
    #These should be the same as those produced by plotbundles.py
    pl.clf()
    for bundle in np.sum(phigreater60_bundled_lpgbthists_allevents,axis=0):
        pl.step((inclusive_hists[1])[:-1], bundle, where='post')
        print ((inclusive_hists[1])[:-1] , bundle, where='post')
    pl.ylim((0,1100000))
    pl.savefig( outdir + "/phiGreater60Integrated.png" )
    pl.clf()
    for bundle in np.sum(philess60_bundled_lpgbthists_allevents,axis=0):
        pl.step((inclusive_hists[1])[:-1], bundle, where='post')
    pl.ylim((0,1100000))
    pl.savefig( outdir + "/phiLess60Integrated.png" )

    
def main():

    try:
        config_file = sys.argv[1]
    except IndexError:
         print ("Please give valid config file")
         exit()
    try:
        with open(config_file,'r') as file:
            config = yaml.load(file,Loader=yaml.FullLoader)
    except EnvironmentError:
        print ("Please give valid config file")
        exit()
        
    #Code to process the input root file,
    #and to get the bundle R/Z histograms per event

    if (config['function']['checkFluctuations']):
        correctionConfig = None
        if 'corrections' in config.keys():
            correctionConfig = config['corrections']

        subconfig = config['checkFluctuations']
        checkFluctuations(initial_state=subconfig['initial_state'], cmsswNtuple=subconfig['cmsswNtuple'], mappingFile=subconfig['mappingFile'], outputName=subconfig['outputName'], correctionConfig = correctionConfig, phisplitConfig = subconfig['phisplit'], beginEvent = subconfig['beginEvent'], endEvent = subconfig['endEvent'])

    #Plotting functions
    
    if (config['function']['plot_MeanMax']):
        subconfig = config['plot_MeanMax']
        plotMeanMax(eventData = subconfig['eventData'], outdir = config['output_dir'], includePhi60 = subconfig['includePhi60'])

    if (config['function']['plot_Truncation']):
        subconfig = config['plot_Truncation']
        plotTruncation(eventData = subconfig['eventData'],outdir = config['output_dir'], includePhi60 = subconfig['includePhi60'] )
        
    if (config['function']['studyTruncationOptions']):
        subconfig = config['studyTruncationOptions']
        studyTruncationOptions(eventData = subconfig['eventData'],outdir = config['output_dir'], includePhi60 = subconfig['includePhi60'] )
        
    
main()
