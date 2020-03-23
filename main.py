#!/usr/bin/env python3
import ROOT
import numpy as np
import mlrose

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from process import getModuleHists,getlpGBTHists,getMinilpGBTGroups,getBundles,getGroupedlpgbtHists,calculateChiSquared
from process import loadDataFile,getTCsPassing,getlpGBTLoadInfo
from plotting import plot, plot2D
from bestchi2 import bestsofar

chi2_min = 50000000000000000000000
combbest = []

def plot_lpGBTLoads(MappingFile,CMSSW_Silicon,CMSSW_Scintillator):

    #Load external data
    data = loadDataFile(MappingFile) #dataframe    
    data_tcs_passing,data_tcs_passing_scin = getTCsPassing(CMSSW_Silicon,CMSSW_Scintillator) #from CMSSW
    
    lpgbt_loads_tcs,lpgbt_loads_words,lpgbt_layers = getlpGBTLoadInfo(data,data_tcs_passing,data_tcs_passing_scin)

    plot(lpgbt_loads_tcs,"loads_tcs.png",binwidth=0.1,xtitle='Number of TCs on a single lpGBT')
    plot(lpgbt_loads_words,"loads_words.png",binwidth=0.1,xtitle='Number of words on a single lpGBT')
    plot2D(lpgbt_loads_tcs,lpgbt_layers,"tcs_vs_layer.png",xtitle='Number of TCs on a single lpGBT')
    plot2D(lpgbt_loads_words,lpgbt_layers,"words_vs_layer.png",xtitle='Number of words on a single lpGBT')

def plot_ModuleLoads(MappingFile,CMSSW_Silicon,CMSSW_Scintillator):

    #Load external data
    data = loadDataFile(MappingFile) #dataframe    
    data_tcs_passing,data_tcs_passing_scin = getTCsPassing(CMSSW_Silicon,CMSSW_Scintillator) #from CMSSW
    
    lpgbt_loads_tcs,lpgbt_loads_words,lpgbt_layers = getlpGBTLoadInfo(data,data_tcs_passing,data_tcs_passing_scin)

    plot(lpgbt_loads_tcs,"loads_tcs.png",binwidth=0.1,xtitle='Number of TCs on a single lpGBT')
    plot(lpgbt_loads_words,"loads_words.png",binwidth=0.1,xtitle='Number of words on a single lpGBT')
    plot2D(lpgbt_loads_tcs,lpgbt_layers,"tcs_vs_layer.png",xtitle='Number of TCs on a single lpGBT')
    plot2D(lpgbt_loads_words,lpgbt_layers,"words_vs_layer.png",xtitle='Number of words on a single lpGBT')

def check_for_missing_modules(MappingFile,CMSSW_Silicon,CMSSW_Scintillator):

    #Load external data
    data = loadDataFile(MappingFile) #dataframe    
    data_tcs_passing,data_tcs_passing_scin = getTCsPassing(CMSSW_Silicon,CMSSW_Scintillator) #from CMSSW
    
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


    
def study_mapping(MappingFile,CMSSW_ModuleHists,algorithm="random_hill_climb",OutputRootFile=False):

    #Load external data
    data = loadDataFile(MappingFile) #dataframe    
    inclusive_hists,module_hists = getModuleHists(CMSSW_ModuleHists)

    #Form hists corresponding to each lpGBT from module hists
    lpgbt_hists = getlpGBTHists(data, module_hists)
    minigroups,minigroups_swap = getMinilpGBTGroups(data)

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
    fitness_cust = mlrose.CustomFitness(mapping_max)
    # Define optimization problem object
    problem_cust = mlrose.DiscreteOpt(length = len(init_state), fitness_fn = fitness_cust, maximize = False, max_val = 1554)

    # Define decay schedule
    schedule = mlrose.ExpDecay()

    if ( OutputRootFile ):
        #Save best combination so far into a root file    
        bundles = getBundles(minigroups,minigroups_swap,bestsofar)
        grouped_hists = getGroupedlpgbtHists(lpgbt_hists,bundles,root=OutputRootFile)
        newfile = ROOT.TFile("lpgbt_8.root","RECREATE")
        for sector in grouped_hists:
            for key, value in sector.items():
                value.Write()
        for sector in inclusive_hists:
            sector.Scale(1./24.)
            sector.Write()
        newfile.Close()
    else:
        if (algorithm == "random_hill_climb"):
            best_state, best_fitness = mlrose.random_hill_climb(problem_cust, max_attempts=10000, max_iters=10000000, restarts=0, init_state=init_state, random_state=1)
            print (best_state)
        elif (algorithm == "genetic_alg"):
            best_state, best_fitness = mlrose.genetic_alg(problem_cust, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=10000, curve=False, random_state=1)
        elif (algorithm == "simulated_annealing"):
            best_state, best_fitness = mlrose.simulated_annealing(problem_cust, schedule = schedule, 
                                                   max_attempts = 100000, max_iters = 10000000, 
                                                   init_state = init_state, random_state = 1)
        else:
            print("Algorithm "+ algorithm + " not known" )


    
def main():

    #Customisation
    MappingFile = "data/FeMappingV7.txt"
    CMSSW_ModuleHists = "data/ROverZHistograms.root"

    #V11
    CMSSW_Silicon = "data/average_tcs_sil_v11_ttbar_20200305.csv"
    CMSSW_Scintillator = "data/average_tcs_scin_v11_ttbar_20200305.csv"
    CMSSW_ModuleHists = "data/ROverZHistograms.root"
    
    #V10
    CMSSW_Silicon_v10 = "data/average_tcs_sil_v10_qg_20200305.csv"
    CMSSW_Scintillator_v10 = "data/average_tcs_scin_v10_qg_20200305.csv"

    study_mapping(MappingFile,CMSSW_ModuleHists,algorithm="random_hill_climb",OutputRootFile=False)

    
    #check_for_missing_modules(MappingFile,CMSSW_Silicon,CMSSW_Scintillator)
    #plot_lpGBTLoads(MappingFile,CMSSW_Silicon,CMSSW_Scintillator)
    #plot_ModuleLoads(MappingFile,CMSSW_Silicon,CMSSW_Scintillator)
    
main()
