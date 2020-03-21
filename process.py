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

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from bestchi2 import bestsofar

np.set_printoptions(threshold=sys.maxsize)

# Define decay schedule
schedule = mlrose.ExpDecay()

pd.set_option('display.max_rows', None)
infiles = []
chi2_min = 47537985399400000000000
#chi2_min = 47537985399400000
combbest = []

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
    
def getMacrolpGBTGroups(minigroups,combination):

    macrogroups = {}
    macrogroup_counter = 0
    maxsize = 67



    #Simplest implementation
    for i in sorted(minigroups.keys()):#loop over minigroups
    #for i in range (len(minigroups)):

        if macrogroup_counter in macrogroups:#if the big group of lpgbts exists

            if ( len(macrogroups[macrogroup_counter]) < (maxsize-len(minigroups[i]) ) ):#if the big group has enough space to accept the next minigroup
                macrogroups[macrogroup_counter].extend( minigroups[i] )
            else: #otherwise start a new big group
                break#temporary
                macrogroup_counter+=1
                macrogroups[macrogroup_counter] = minigroups[i].copy()
        else:
            macrogroups[macrogroup_counter] = minigroups[i].copy()


    return macrogroups

def getBundles(minigroups,minigroups_swap,combination):

    bundles = np.array_split(combination,24)

    for bundle in bundles:
        for lpgbt in minigroups_swap[minigroups[bundle[-1]]]:
            if not lpgbt in bundle[-5:]:
                np.append(bundle,lpgbt)
    #Check the last lpgbt of the bundle, does it belong to a mini-group?    

    
    return bundles
            
def getGroupedlpgbtHists(hists,groups,root=False):

    grouped_lpgbthists = []
    grouped_lpgbthists_list = []

    for p,phiselection in enumerate(hists):

        #temp = {}
        temp_list = {}

        #for i in sorted(groups.keys()):#loop over groups
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


def getGroupedlpgbtHistsOneBundle(hists,groups):

    grouped_lpgbthists = []
    grouped_lpgbthists_list = []

    for p,phiselection in enumerate(hists):

        lpgbt_hist = ROOT.TH1D( ("lpgbt_ROverZ_grouped_" + str(p)),"",42,0.076,0.518);
        lpgbt_hist_list = [] 
            
        for lpgbt in groups:#loop over each lpgbt in the big group
            lpgbt_hist.Add( phiselection[lpgbt]  )
 
        for b in range(1,lpgbt_hist.GetNbinsX()+1): 
            lpgbt_hist_list.append(lpgbt_hist.GetBinContent(b))

        grouped_lpgbthists_list.append(lpgbt_hist_list)

    return grouped_lpgbthists_list



def calculateChiSquared(inclusive,grouped):

    chi2_total = 0

    for i in range(2):

        for key,hist in grouped[i].items():

            for b in range(len(hist)):

                squared_diff = np.power(hist[b]-inclusive[i].GetBinContent(b+1)/24, 2 )   

                chi2_total+=squared_diff

    return chi2_total

def calculateChiSquaredOneBundle(inclusive,grouped):

    chi2_total = 0

    for i,hist in enumerate(grouped):

        for b in range(len(hist)):

            squared_diff = np.power(hist[b]-inclusive[i].GetBinContent(b+1)/24, 2 )   

            chi2_total+=squared_diff

    return chi2_total

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

def generate_groups(lst, n):
    if not lst:
        yield []
    else:
        for group in (((lst[0],) + xs) for xs in itertools.combinations(lst[1:], n-1)):
            for groups in generate_groups([x for x in lst if x not in group], n):
                yield [group] + groups

def main():

    #Customisation
    MappingFile = "data/FeMappingV7.txt"

    #V11
    CMSSW_Silicon = "data/average_tcs_sil_v11_ttbar_20200305.csv"
    CMSSW_Scintillator = "data/average_tcs_scin_v11_ttbar_20200305.csv"
    CMSSW_ModuleHists = "data/ROverZHistograms.root"
    
    #V10
    # CMSSW_Silicon = "data/average_tcs_sil_v10_qg_20200305.csv"
    # CMSSW_Scintillator = "data/average_tcs_scin_v10_qg_20200305.csv"

    
    Plot_lpGBTLoads = False
    Plot_ModuleLoads = False
    study_mapping = True
    
    
    #Load Data    
    data = loadDataFile(MappingFile) #dataframe
    data_tcs_passing,data_tcs_passing_scin = getTCsPassing(CMSSW_Silicon,CMSSW_Scintillator) #from CMSSW


    #print (module_hists[0].Integral(1,1), module_hists[0].GetName())


    #Check for missing modules
    #check_for_missing_modules(data,data_tcs_passing,data_tcs_passing_scin)



    # for key in minigroups:
    #     print(key, '->', minigroups[key])
        

    
    if ( study_mapping ):
        inclusive_hists,module_hists = getModuleHists(CMSSW_ModuleHists)
        lpgbt_hists = getlpGBTHists(data, module_hists)
        minigroups,minigroups_swap = getMinilpGBTGroups(data)


        
        #Print bestsofar in root file
        # mg = getBundles(minigroups,minigroups_swap,bestsofar)
        # gh = getGroupedlpgbtHists(lpgbt_hists,mg,root=True)
        # newfile = ROOT.TFile("lpgbt_6.root","RECREATE")
        # for sector in gh:
        #     for key, value in sector.items():
        #         value.Write()
        # for sector in inclusive_hists:
        #     sector.Scale(1./24.)
        #     sector.Write()
        
        # newfile.Close()
        
        
        def mapping_max(state):
            global chi2_min
            global combbest

            chi2 = 0

            macrogroups = getBundles(minigroups,minigroups_swap,state)
            grouped_lpgbthists = getGroupedlpgbtHists(lpgbt_hists,macrogroups)
            chi2 = calculateChiSquared(inclusive_hists,grouped_lpgbthists)



            if (chi2<chi2_min):
                chi2_min = chi2
                combbest = np.copy(state)
                print (chi2_min)
                print (repr(combbest))
                
            return chi2

        init_state = bestsofar
        #print (init_state)
        fitness_cust = mlrose.CustomFitness(mapping_max)
        # Define optimization problem object
        problem_cust = mlrose.DiscreteOpt(length = len(init_state), fitness_fn = fitness_cust, maximize = False, max_val = 1554)

        best_state, best_fitness = mlrose.random_hill_climb(problem_cust, max_attempts=10000, max_iters=10000000, restarts=0, init_state=init_state, random_state=1)
        #best_state, best_fitness = mlrose.genetic_alg(problem_cust, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=10000, curve=False, random_state=1)
        # best_state, best_fitness = mlrose.simulated_annealing(problem_cust, schedule = schedule, 
        #                                               max_attempts = 100000, max_iters = 10000000, 
        #                                               init_state = init_state, random_state = 1)

        
        #print (best_state)
         #2996258530

         #253347326517


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
