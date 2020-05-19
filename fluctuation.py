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
    

# def getlpGBTHistsNumpy(data, module_hists):

#     lpgbt_hists = []

#     for p,phiselection in enumerate(module_hists):#inclusive and phi < 60

#         temp = {}

#         for lpgbt in range(0,1600) :
#             lpgbt_found = False

#             #lpgbt_hist = ROOT.TH1D( ("lpgbt_ROverZ_silicon_" + str(lpgbt) + "_" + str(p)),"",42,0.076,0.58);
#             lpgbt_hist = np.zeros(42)
            
#             for tpg_index in ['TPGId1','TPGId2']:#lpgbt may be in the first or second position in the file

#                 for index, row in (data[data[tpg_index]==lpgbt]).iterrows():  
#                     lpgbt_found = True
                    
#                     if (row['density']==2):#Scintillator

#                         hist = phiselection[ 1, row['u'], row['v'], row['layer'] ] # get module hist
#                     else:
#                         hist = phiselection[ 0, row['u'], row['v'], row['layer'] ] # get module hist        

#                     linkfrac = 'TPGeLinkFrac1'
#                     if ( tpg_index == 'TPGId2' ):
#                         linkfrac = 'TPGeLinkFrac2'

#                     #lpgbt_hist.Add( hist, row[linkfrac] ) # add module hist with the correct e-link weight
#                     weighted = hist*row[linkfrac]
#                     lpgbt_hist = lpgbt_hist + weighted
                    
#             if lpgbt_found:
#                 temp[lpgbt] = lpgbt_hist

#         lpgbt_hists.append(temp)
    
#     return lpgbt_hists

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




def checkFluctuations(initial_state, cmsswNtuple, dataFile):

    #List of which minigroups are assigned to each bundle 
    init_state = np.hstack(np.load(initial_state,allow_pickle=True))

    #Load the CMSSW ntuple to get per event and per trigger cell information
    rootfile = ROOT.TFile.Open( cmsswNtuple , "READ" )
    tree = rootfile.Get("HGCalTriggerNtuple")

    #Load mapping file
    data = loadDataFile(dataFile) 

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
                    uv = rotate_to_sector_0(u,v,layer)
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

            #Sum the individual module histograms to get the minigroup histograms
            module_hists_inc = {}
            module_hists_phi60 = {}
            inclusive_hists = np.histogram( ROverZ_Inclusive, bins = 42, range = (0.076,0.58) )
            inclusive_hists_phi60 = np.histogram( ROverZ_Inclusive_Phi60, bins = 42, range = (0.076,0.58) )

            for key,value in ROverZ_per_module.items():
                module_hists_inc[key] = np.histogram( value, bins = 42, range = (0.076,0.58) )[0]
            for key,value in ROverZ_per_module_Phi60.items():
                module_hists_phi60[key] = np.histogram( value, bins = 42, range = (0.076,0.58) )[0]


            module_hists = [module_hists_inc,module_hists_phi60]

            #Should be able to get a list of modules for each minigroup, and then just sum those to save time.

            # lpgbt_hists = getlpGBTHistsNumpy(data,module_hists)


            minigroup_hists = getMiniGroupHistsNumpy(module_hists,minigroups_modules)


            bundled_lpgbthists = getBundledlpgbtHists(minigroup_hists,bundles)

            bundled_lpgbthists_allevents.append(bundled_lpgbthists)




    except KeyboardInterrupt:
        print("interrupt received, stopping and saving")

    finally:

        with open("alldata.txt", "wb") as filep:   #Pickling
            pickle.dump(bundled_lpgbthists_allevents, filep)



def analyseFluctuations():
    with open("alldata_1528.txt", "rb") as filep:   #Pickling
        bundled_lpgbthists_allevents = pickle.load(filep)

    inclusive_hists = np.histogram( np.empty(0), bins = 42, range = (0.076,0.58) )

    
    inclusive = 0


    #Plotting Max, mean and standard deviation per bundle:
    plotMaxMean = False
    if ( plotMaxMean ):
        for bundle in range(24):

            list_over_events = []
            for event in bundled_lpgbthists_allevents:
                list_over_events.append( event[inclusive][bundle] )

            hist_max = np.maximum.reduce(list_over_events)
            hist_mean = np.mean(list_over_events, axis=0)
            hist_std = np.std(list_over_events, axis=0)

            for s,std in enumerate(hist_std):
                hist_std[s] = std + hist_mean[s]

            pl.bar((inclusive_hists[1])[:-1], hist_max, width=0.012)
            pl.bar((inclusive_hists[1])[:-1], hist_std, width=0.012)
            pl.bar((inclusive_hists[1])[:-1], hist_mean, width=0.012)


            # for e,event in enumerate(list_over_events):
            #     pl.bar((inclusive_hists[1])[:-1], event, width=0.012,fill=False)
            #     #if (e>200): break


            pl.savefig( "plots/bundle_" + str(bundle) + "max.png" )
            pl.clf()




        
    # for entry,event in enumerate(bundled_lpgbthists_allevents):
    #     for key,bundle in event[inclusive].items():
    #         #print (bundle)
    #         pl.bar((inclusive_hists[1])[:-1], bundle, width=0.012)
    #         pl.savefig( "plots/entry_" + str(entry) + "silicon_" + str(key) + ".png" )
    #         pl.clf()

    # for hist in hists_max:
    #     pl.bar((inclusive_hists[1])[:-1], hist, width=0.012)
    # pl.savefig( "plots/maxima.png" )
    # pl.clf()



    #Loop over all events to get the maximum per bundle

    hists_max = []
    for bundle in range(24):
        list_over_events = []
        for event in bundled_lpgbthists_allevents:
            list_over_events.append( event[inclusive][bundle] )
        hists_max.append( np.maximum.reduce(list_over_events) )

    #Find the maximum per bin over all events,
    #Then multiply this by 0.99 for a 1% truncation
    #maxima_hists = []

    overall_max = np.amax(hists_max, axis=0)
    #print ("max",overall_max)
    overall_max99 = np.round(overall_max*0.99)
    overall_max95 = np.round(overall_max*0.95)
    overall_max90 = np.round(overall_max*0.90)

    #Loop back over events, counting the maximum wait time
    #for each bin, with and without truncation
    maximum_per_event = []
    maximum_per_event99 = []
    maximum_per_event95 = []
    maximum_per_event90 = []
    
    for event in bundled_lpgbthists_allevents:

        bundle_hists = np.array(list(event[inclusive].values()))
        maximum = np.amax(bundle_hists, axis=0)
        maximum99 = np.where( np.less( overall_max99, maximum ), overall_max99, maximum )
        maximum95 = np.where( np.less( overall_max95, maximum ), overall_max95, maximum )
        maximum90 = np.where( np.less( overall_max90, maximum ), overall_max90, maximum )
        
        maximum_per_event.append( np.sum(maximum) )
        maximum_per_event99.append( np.sum(maximum99) )
        maximum_per_event95.append( np.sum(maximum95) )
        maximum_per_event90.append( np.sum(maximum90) )


        #print ( "fullmax",maximum  )
        #print ( "99",maximum99  )

    #print (np.array(maximum_per_event)-np.array(maximum_per_event99))
    #print (np.array(maximum_per_event)-np.array(maximum_per_event95))

    pl.hist(np.array(maximum_per_event)-np.array(maximum_per_event99),40,(0,40),histtype='step',log=True,label='1% truncation')
    pl.hist(np.array(maximum_per_event)-np.array(maximum_per_event95),40,(0,40),histtype='step',log=True,label='5% truncation')
    pl.hist(np.array(maximum_per_event)-np.array(maximum_per_event90),40,(0,40),histtype='step',log=True,label='10% truncation')    
    pl.xlabel('Number of TCs truncated')
    pl.ylabel('Number of Events')
    pl.legend()
    pl.savefig( "plots/truncation.png" )
    
def main():

    
    #Code to process the input root file,
    #and to get the bundle R/Z histograms per event

    runCheckFluctuations = True
    initial_state = "bundles_job_202.npy"
    cmsswNtuple = "data/small_v11_neutrino_gun_200415.root"
    dataFile = "data/FeMappingV7.txt"
    if (runCheckFluctuations):
        checkFluctuations(initial_state=initial_state, cmsswNtuple=cmsswNtuple, dataFile=dataFile)
    

    runAnalyseFluctuations = False
    if (runAnalyseFluctuations):
        analyseFluctuations()

main()
