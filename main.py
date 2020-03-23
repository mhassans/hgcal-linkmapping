#!/usr/bin/env python3
import ROOT
import mlrose
import numpy as np
from process import getModuleHists,getlpGBTHists,getMinilpGBTGroups,getBundles,getGroupedlpgbtHists,loadDataFile,calculateChiSquared
from bestchi2 import bestsofar

chi2_min = 50000000000000000000000
combbest = []

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

    MappingFile = "data/FeMappingV7.txt"
    CMSSW_ModuleHists = "data/ROverZHistograms.root"


    study_mapping(MappingFile,CMSSW_ModuleHists,algorithm="random_hill_climb",OutputRootFile=False)

main()
