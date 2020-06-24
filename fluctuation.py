#!/usr/bin/env python3
import ROOT
import numpy as np
import matplotlib.pyplot as pl
import math
import pickle
from process import loadDataFile
from process import getMinilpGBTGroups,getBundles,getBundledlpgbtHists
from rotate import rotate_to_sector_0
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

def getROverZPhi(x, y, z):

    if (z < 0):
        x = x*-1
    
    r = math.sqrt( x*x + y*y  );
    phi = np.arctan2(y,x);

    phi = phi + np.pi;
    if ( phi < (2*np.pi/3) ):
        phi = phi;
    elif ( phi < (4*np.pi/3) ):
        phi = phi-(2*np.pi/3);
    else:
        phi = phi-(4*np.pi/3);

    roverz_phi = [r/z,phi]
    return roverz_phi;

def etaphiMapping(layer, etaphi):

    if (etaphi[1] <= 48):
        sector = 0
    elif (etaphi[1] > 48 and etaphi[1] <= 96):
        sector = 1
    else:
        sector = 2

    if (sector==1):
        pp=etaphi[1]-48
    elif(sector==2):
        pp=etaphi[1]-96
    else:
        pp = etaphi[1]
  
    pp = (pp-1)//4;# //Phi index 1-12
  
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

    return [ep,pp]




def checkFluctuations(initial_state, cmsswNtuple, mappingFile, outputName="alldata"):

    #List of which minigroups are assigned to each bundle 
    init_state = np.hstack(np.load(initial_state,allow_pickle=True))

    #Load the CMSSW ntuple to get per event and per trigger cell information
    rootfile = ROOT.TFile.Open( cmsswNtuple , "READ" )
    tree = rootfile.Get("HGCalTriggerNtuple")

    #Load mapping file
    data = loadDataFile(mappingFile) 

    #Get list of which lpgbts are in each minigroup
    minigroups,minigroups_swap = getMinilpGBTGroups(data)

    #Get list of which modules are in each minigroup
    minigroups_modules = getMiniModuleGroups(data,minigroups_swap)
    bundles = getBundles(minigroups_swap,init_state)

    bundled_lpgbthists_allevents = []
    
    ROverZ_per_module = {}
    ROverZ_per_module_Phi60 = {}
    
    for i in range (15):
        for j in range (15):
            for k in range (1,53):
                if  k < 28 and k%2 == 0:
                    continue
                ROverZ_per_module[0,i,j,k] = np.empty(0)
                ROverZ_per_module_Phi60[0,i,j,k] = np.empty(0)

    for i in range (5):
        for j in range (12):
            for k in range (37,53):
                ROverZ_per_module[1,i,j,k] = np.empty(0)
                ROverZ_per_module_Phi60[1,i,j,k] = np.empty(0)

    try:
        for entry,event in enumerate(tree):
            # if entry > 10:
            #     break
            print ("Event number " + str(entry))
    
            for key in ROverZ_per_module.keys():
                ROverZ_per_module[key] = np.empty(0)
                ROverZ_per_module_Phi60[key] = np.empty(0)

            #Loop over list of trigger cells in a particular
            #event and fill R/Z histograms for each module
            #(inclusively and for phi < 60)
            
            for u,v,layer,x,y,z,cellu,cellv in zip(event.tc_waferu,event.tc_waferv,event.tc_layer,event.tc_x,event.tc_y,event.tc_z,event.tc_cellu,event.tc_cellv):

                eta_phi = getROverZPhi(x,y,z)

                if ( u > -990 ): #Silicon
                    uv,sector = rotate_to_sector_0(u,v,layer)
                    ROverZ_per_module[0,uv[0],uv[1],layer] = np.append(ROverZ_per_module[0,uv[0],uv[1],layer],abs(eta_phi[0]))            
                    if (eta_phi[1] < np.pi/3):
                        ROverZ_per_module_Phi60[0,uv[0],uv[1],layer] = np.append(ROverZ_per_module_Phi60[0,uv[0],uv[1],layer],abs(eta_phi[0]))
                        
                else: #Scintillator  
                    eta = cellu
                    phi = cellv
                    etaphi = etaphiMapping(layer,[eta,phi]);
                    ROverZ_per_module[1,etaphi[0],etaphi[1],layer] = np.append(ROverZ_per_module[1,etaphi[0],etaphi[1],layer],abs(eta_phi[0]))            
                    if (eta_phi[1] < np.pi/3):
                        ROverZ_per_module_Phi60[1,etaphi[0],etaphi[1],layer] = np.append(ROverZ_per_module_Phi60[1,etaphi[0],etaphi[1],layer],abs(eta_phi[0]))


            ROverZ_Inclusive = np.empty(0)
            ROverZ_Inclusive_Phi60 = np.empty(0)

            for key,value in ROverZ_per_module.items():
                ROverZ_Inclusive = np.append(ROverZ_Inclusive,value)
            for key,value in ROverZ_per_module_Phi60.items():
                ROverZ_Inclusive_Phi60 = np.append(ROverZ_Inclusive_Phi60,value)

            #Bin the TC module data
            module_hists_inc = {}
            module_hists_phi60 = {}
            inclusive_hists = np.histogram( ROverZ_Inclusive, bins = 42, range = (0.076,0.58) )
            inclusive_hists_phi60 = np.histogram( ROverZ_Inclusive_Phi60, bins = 42, range = (0.076,0.58) )

            for key,value in ROverZ_per_module.items():
                module_hists_inc[key] = np.histogram( value, bins = 42, range = (0.076,0.58) )[0]
            for key,value in ROverZ_per_module_Phi60.items():
                module_hists_phi60[key] = np.histogram( value, bins = 42, range = (0.076,0.58) )[0]

            module_hists = [module_hists_inc,module_hists_phi60]

            #Sum the individual module histograms to get the minigroup histograms
            minigroup_hists = getMiniGroupHistsNumpy(module_hists,minigroups_modules)

            #Sum the minigroup histograms to get the bundle histograms
            bundled_lpgbthists = getBundledlpgbtHists(minigroup_hists,bundles)

            bundled_lpgbthists_allevents.append(bundled_lpgbthists)


    except KeyboardInterrupt:
        print("interrupt received, stopping and saving")

    finally:

        #Write all data to file for later analysis (Pickling)
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
            list_over_events = list_over_events_inclusive
        else:
            list_over_events = list_over_events_maximum

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
        
        pl.savefig( outdir + "/bundle_" + str(bundle) + "max.png" )
        pl.clf()

        hists_max.append(hist_max)
        
    #Plot maxima for all bundles on the same plot
    for hist in hists_max:
        pl.bar((inclusive_hists[1])[:-1], hist, width=0.012,align='edge')
    pl.xlabel('r/z')
    pl.ylabel('Maximum number of TCs per bin')
    pl.savefig( outdir + "/maxima.png" )
    pl.clf()


def plotTruncation(eventData, outdir = ".", includePhi60 = True):
    #Load pickled per-event bundle histograms
    with open(eventData, "rb") as filep:   
        bundled_lpgbthists_allevents = pickle.load(filep)
    os.system("mkdir -p " + outdir)

    nbins = 42
    nbundles = 24
    
    #To get binning for r/z histograms
    inclusive_hists = np.histogram( np.empty(0), bins = 42, range = (0.076,0.58) )

    #Names for inclusive and phi < 60 indices
    inclusive = 0
    phi60 = 1
    
    #Loop over all events to get the maximum per bin over bundles (per event)
    
    inclusive_bundled_lpgbthists_allevents = np.empty((len(bundled_lpgbthists_allevents),nbundles,nbins))
    phi60_bundled_lpgbthists_allevents = np.empty((len(bundled_lpgbthists_allevents),nbundles,nbins))
    
    for e,event in enumerate(bundled_lpgbthists_allevents):
        inclusive_bundled_lpgbthists_allevents[e] = np.array(list(event[inclusive].values()))
        phi60_bundled_lpgbthists_allevents[e] = np.array(list(event[phi60].values()))

    #Form the intersection of the inclusive and phi60 arrays,
    #taking for each bin the maximum of the inclusive and phi60 x 2

    maximum_bundled_lpgbthists_allevents = np.maximum(inclusive_bundled_lpgbthists_allevents,phi60_bundled_lpgbthists_allevents*2)
    
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
    
    for event in bundled_lpgbthists_allevents:

        bundle_hists_inclusive = np.array(list(event[inclusive].values()))
        bundle_hists_phi60 = np.array(list(event[phi60].values()))

        bundle_hists_maximum = np.maximum(bundle_hists_inclusive,bundle_hists_phi60*2)
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
    
    print ("Maximum TC in any bundle in any event (per bin) = ", np.round(np.amax(max_per_event_perbin,axis=0)/6))
    print ("Sum of per-bin maximum TC (over bundles and events) = ",  np.round(np.sum(np.amax(max_per_event_perbin,axis=0)/6)))
    print ("Sum of per-bin maximum TC (over bundles and events) with 1% truncation =", np.round(np.sum(np.amax(max_per_event_perbin99,axis=0)/6)))
    print ("Sum of per-bin maximum TC (over bundles and events) with 5% truncation = ", np.round(np.sum(np.amax(max_per_event_perbin95,axis=0)/6)))
    print ("Sum of per-bin maximum TC (over bundles and events) with 10% truncation = ", np.round(np.sum(np.amax(max_per_event_perbin90,axis=0)/6)))
        
    pl.hist(np.sum(np.array(total_per_event)-np.array(total_per_event99),axis=1)/(6*24),50,(0,5),histtype='step',log=True,label='1% truncation')
    pl.hist(np.sum(np.array(total_per_event)-np.array(total_per_event95),axis=1)/(6*24),50,(0,5),histtype='step',log=True,label='5% truncation')
    pl.hist(np.sum(np.array(total_per_event)-np.array(total_per_event90),axis=1)/(6*24),50,(0,5),histtype='step',log=True,label='10% truncation')    
    pl.xlabel('Number of TCs truncated on average per bundle')
    # pl.hist(np.sum(np.array(total_per_event)-np.array(total_per_event99),axis=1)/(6),50,(0,100),histtype='step',log=True,label='1% truncation')
    # pl.hist(np.sum(np.array(total_per_event)-np.array(total_per_event95),axis=1)/(6),50,(0,100),histtype='step',log=True,label='5% truncation')
    # pl.hist(np.sum(np.array(total_per_event)-np.array(total_per_event90),axis=1)/(6),50,(0,100),histtype='step',log=True,label='10% truncation')    
    # pl.xlabel('Number of TCs truncated')
    
    pl.ylabel('Number of Events')
    pl.legend()
    pl.savefig( outdir + "/truncation.png" )

    pl.clf()
    pl.step((inclusive_hists[1])[:-1], ratio_to_best)
    pl.axhline(y=1, color='r', linestyle='--')
    pl.xlabel('r/z')
    pl.ylabel('Ratio of 1% truncation to likely best')
    pl.ylim((0,10))
    pl.savefig( outdir + "/ratio_to_best.png" )
    
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
        subconfig = config['checkFluctuations']
        checkFluctuations(initial_state=subconfig['initial_state'], cmsswNtuple=subconfig['cmsswNtuple'], mappingFile=subconfig['mappingFile'], outputName=subconfig['outputName'])

    #Plotting functions
    
    if (config['function']['plot_MeanMax']):
        subconfig = config['plot_MeanMax']
        plotMeanMax(eventData = subconfig['eventData'], outdir = config['output_dir'], includePhi60 = subconfig['includePhi60'])

    if (config['function']['plot_Truncation']):
        subconfig = config['plot_Truncation']
        plotTruncation(eventData = subconfig['eventData'],outdir = config['output_dir'], includePhi60 = subconfig['includePhi60'] )
        
    
main()
