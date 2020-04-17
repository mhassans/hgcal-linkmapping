#!/usr/bin/env python3
import ROOT
import numpy as np
import matplotlib.pyplot as pl
import math
from process import loadDataFile
from process import getMinilpGBTGroups,getBundles,getBundledlpgbtHists,getlpGBTHists
from rotate import rotate_to_sector_0

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

    for i in range (15):
        for j in range (15):
            for k in range (1,53):
                if  k < 28 and k%2 == 0:
                    continue
                ROverZ_per_module[0,i,j,k] = np.empty(0)
    
    for entry,event in enumerate(tree):
        if entry > 0:
            break

        for u,v,layer,x,y,z in zip(event.tc_waferu,event.tc_waferv,event.tc_layer,event.tc_x,event.tc_y,event.tc_z):

            eta_phi = getROverZPhi(x,y,z)
            
            if ( u > -990 ):
                uv = rotate_to_sector_0(u,v,layer)


                ROverZ_per_module[0,uv[0],uv[1],layer] = np.append(ROverZ_per_module[0,uv[0],uv[1],layer],abs(eta_phi[0]))            
                #ROverZ_per_module[0,uv[0],uv[1],layer].append( abs(eta_phi[0] ) )


        hists = {}
        for i in range (15):
            for j in range (15):
                for k in range (1,53):
                    if  k < 28 and k%2 == 0:
                        continue
                    #if ( ROverZ_per_module[0,i,j,k].size > 0 ):
                        #hists = pl.hist( ROverZ_per_module[0,i,j,k], bins = 42, range = (0.076,0.58) )
                        #pl.savefig( "plots/silicon_" + str(i) + "_" + str(j) + "_" + str(k) + ".png" )
                        #pl.clf()
                    hists[0,i,j,k] = np.histogram( ROverZ_per_module[0,i,j,k], bins = 42, range = (0.076,0.58) )
                    #hists[0,i,j,k] = pl.hist( ROverZ_per_module[0,i,j,k], bins = 42, range = (0.076,0.58) )
                    #pl.clf()
                    #print (hists[0,i,j,k])
                    #print (ROverZ_per_module[0,i,j,k])




        for key,value in hists.items():
            #print ("key = ",key)
            #print ("value = ",value)
            if (ROverZ_per_module[0,key[1],key[2],key[3]].size > 0 ):
                #pl.hist( value )
                pl.bar((value[1])[:-1], value[0], width=0.012)
                pl.savefig( "plots/silicon_" + str(key[1]) + "_" + str(key[2]) + "_" + str(key[3]) + ".png" )
                pl.clf()
            
        # inclusive_hists,module_hists = getModuleHists(CMSSW_ModuleHists)
        # lpgbt_hists = getlpGBTHists(data, module_hists)



checkFluctuations()
