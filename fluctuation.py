#!/usr/bin/env python3
import ROOT
import numpy as np
import matplotlib.pyplot as pl
import math
from process import loadDataFile
from process import getMinilpGBTGroups,getBundles,getBundledlpgbtHists
from rotate import rotate_to_sector_0
import time

def getlpGBTHistsNumpy(data, module_hists):

    lpgbt_hists = []

    for p,phiselection in enumerate(module_hists):#inclusive and phi < 60

        temp = {}

        for lpgbt in range(0,1600) :
            lpgbt_found = False

            #lpgbt_hist = ROOT.TH1D( ("lpgbt_ROverZ_silicon_" + str(lpgbt) + "_" + str(p)),"",42,0.076,0.58);
            lpgbt_hist = np.zeros(42)
            
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

                    #lpgbt_hist.Add( hist, row[linkfrac] ) # add module hist with the correct e-link weight
                    weighted = hist*row[linkfrac]
                    lpgbt_hist = lpgbt_hist + weighted
                    
            if lpgbt_found:
                temp[lpgbt] = lpgbt_hist

        lpgbt_hists.append(temp)
    
    return lpgbt_hists

def getMiniGroupHistsNumpy(lpgbt_hists, minigroups_swap):
    
    minigroup_hists = []

    minigroup_hists_inclusive = {}
    minigroup_hists_phi60 = {}

    for minigroup, lpgbts in minigroups_swap.items():
        
        inclusive = np.zeros(42)
        phi60 = np.zeros(42)

        for lpgbt in lpgbts:
            inclusive = inclusive + lpgbt_hists[0][lpgbt]
            # if ( minigroup == 22):
            #     print ( lpgbt,lpgbt_hists[0][lpgbt] )
            #     print ( lpgbt,inclusive )

            phi60 = phi60 + lpgbt_hists[1][lpgbt]

        minigroup_hists_inclusive[minigroup] = inclusive.copy()
        minigroup_hists_phi60[minigroup] = phi60.copy()
        #if ( minigroup == 22):
            #print ( minigroup, minigroup_hists_inclusive[minigroup] )
            
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




def checkFluctuations():

    initial_state = "bundles_job_202.npy"
    init_state = np.hstack(np.load(initial_state,allow_pickle=True))
    rootfile = ROOT.TFile.Open( "data/small_v11_neutrino_gun_200415.root" , "READ" )
    tree = rootfile.Get("HGCalTriggerNtuple")


    #Load external data
    data = loadDataFile("data/FeMappingV7.txt") #dataframe    
    minigroups,minigroups_swap = getMinilpGBTGroups(data)
    bundles = getBundles(minigroups_swap,init_state)
    #    bundled_lpgbthists = getBundledlpgbtHists(minigroup_hists,bundles)

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
                
    for entry,event in enumerate(tree):
        if entry > 10:
            break
        print ("Event number " + str(entry))
        start = time.time()
        for u,v,layer,x,y,z,cellu,cellv in zip(event.tc_waferu,event.tc_waferv,event.tc_layer,event.tc_x,event.tc_y,event.tc_z,event.tc_cellu,event.tc_cellv):

            eta_phi = getROverZPhi(x,y,z)
            
            if ( u > -990 ):
                uv = rotate_to_sector_0(u,v,layer)
                ROverZ_per_module[0,uv[0],uv[1],layer] = np.append(ROverZ_per_module[0,uv[0],uv[1],layer],abs(eta_phi[0]))            
                if (eta_phi[1] < np.pi/3):
                    ROverZ_per_module_Phi60[0,uv[0],uv[1],layer] = np.append(ROverZ_per_module_Phi60[0,uv[0],uv[1],layer],abs(eta_phi[0]))
            else:
                eta = cellu
                phi = cellv
                etaphi = etaphiMapping(layer,[eta,phi]);
                ROverZ_per_module[1,etaphi[0],etaphi[1],layer] = np.append(ROverZ_per_module[1,etaphi[0],etaphi[1],layer],abs(eta_phi[0]))            
                if (eta_phi[1] < np.pi/3):
                    ROverZ_per_module_Phi60[1,etaphi[0],etaphi[1],layer] = np.append(ROverZ_per_module_Phi60[1,etaphi[0],etaphi[1],layer],abs(eta_phi[0]))

                
        end = time.time()
        print("end of TC manip",end - start)

        ROverZ_Inclusive = np.empty(0)
        ROverZ_Inclusive_Phi60 = np.empty(0)

        for key,value in ROverZ_per_module.items():
            ROverZ_Inclusive = np.append(ROverZ_Inclusive,value)
        for key,value in ROverZ_per_module_Phi60.items():
            ROverZ_Inclusive_Phi60 = np.append(ROverZ_Inclusive_Phi60,value)
            
        module_hists_inc = {}
        module_hists_phi60 = {}
        inclusive_hists = np.histogram( ROverZ_Inclusive, bins = 42, range = (0.076,0.58) )
        inclusive_hists_phi60 = np.histogram( ROverZ_Inclusive_Phi60, bins = 42, range = (0.076,0.58) )
        
        for key,value in ROverZ_per_module.items():
            module_hists_inc[key] = np.histogram( value, bins = 42, range = (0.076,0.58) )[0]
        for key,value in ROverZ_per_module_Phi60.items():
            module_hists_phi60[key] = np.histogram( value, bins = 42, range = (0.076,0.58) )[0]

        end = time.time()
        print("made all hists",end - start)

            
        module_hists = [module_hists_inc,module_hists_phi60]

        #Should be able to get a list of modules for each minigroup, and then just sum those to save time.

        lpgbt_hists = getlpGBTHistsNumpy(data,module_hists)

        end = time.time()
        print("got lpgbt hists",end - start)

        
        # for key,val in lpgbt_hists[0].items():
        #      print (key,val)


        minigroup_hists = getMiniGroupHistsNumpy(lpgbt_hists,minigroups_swap)
        
        end = time.time()
        print("got mg hists",end - start)

        # for key,mg in minigroup_hists[0].items():
        #     print (key,mg)
        

        bundled_lpgbthists = getBundledlpgbtHists(minigroup_hists,bundles)
        end = time.time()
        print("got bundled hists",end - start)
        


        
        #print (inclusive_hists[1])
        # for key,bundle in bundled_lpgbthists[0].items():
        #     #print (bundle)
        #     #pl.hist( bundle )
        #     pl.bar((inclusive_hists[1])[:-1], bundle, width=0.012)
        #     pl.savefig( "plots/silicon_" + str(key) + ".png" )
        #     pl.clf()

            
        # for i in range (15):
        #     for j in range (15):
        #         for k in range (1,53):
        #             if  k < 28 and k%2 == 0:
        #                 continue
        #             module_hists[0,i,j,k] = np.histogram( ROverZ_per_module[0,i,j,k], bins = 42, range = (0.076,0.58) )
                    #hists[0,i,j,k] = pl.hist( ROverZ_per_module[0,i,j,k], bins = 42, range = (0.076,0.58) )
                    #pl.clf()
                    #print (hists[0,i,j,k])
                    #print (ROverZ_per_module[0,i,j,k])

        


        # for key,value in hists.items():
        #     #print ("key = ",key)
        #     #print ("value = ",value)
        #     if (ROverZ_per_module[0,key[1],key[2],key[3]].size > 0 ):
        #         #pl.hist( value )
        #         pl.bar((value[1])[:-1], value[0], width=0.012)
        #         pl.savefig( "plots/silicon_" + str(key[1]) + "_" + str(key[2]) + "_" + str(key[3]) + ".png" )
        #         pl.clf()
            
        # inclusive_hists,module_hists = getModuleHists(CMSSW_ModuleHists)
        # lpgbt_hists = getlpGBTHists(data, module_hists)



checkFluctuations()
