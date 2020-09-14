#!/usr/bin/env python3
import sys
sys.path.insert(1, './externals')
import ROOT
import numpy as np
import mlrose_mod as mlrose # Author: Genevieve Hayes https://github.com/gkhayes/mlrose/tree/master/mlrose
import time
import yaml
import signal

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

from process import getModuleHists, getlpGBTHists, getMiniGroupHists, getMinilpGBTGroups, getMiniModuleGroups, getBundles, getBundledlpgbtHists, getBundledlpgbtHistsRoot, calculateChiSquared, getMaximumNumberOfModulesInABundle
from process import loadDataFile, getTCsPassing, getlpGBTLoadInfo, getHexModuleLoadInfo, getModuleTCHists
from plotting import plot, plot2D
from example_minigroup_configuration import example_minigroup_configuration

from geometryCorrections import applyGeometryCorrections

chi2_min = 50000000000000000000000
combbest = []

def handler(signum, frame):
    raise ValueError()    

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
    
    lpgbt_loads_tcs,lpgbt_loads_words,lpgbt_layers = getHexModuleLoadInfo(data,data_tcs_passing,data_tcs_passing_scin)

    plot(lpgbt_loads_tcs,"loads_tcs.png",binwidth=0.1,xtitle='Number of TCs on a single lpGBT')
    plot(lpgbt_loads_words,"loads_words.png",binwidth=0.1,xtitle='Number of words on a single lpGBT')
    plot2D(lpgbt_loads_tcs,lpgbt_layers,"tcs_vs_layer.png",xtitle='Number of TCs on a single lpGBT')
    plot2D(lpgbt_loads_words,lpgbt_layers,"words_vs_layer.png",xtitle='Number of words on a single lpGBT')
    
def produce_AllocationFile(MappingFile,allocation,minigroup_type="minimal"):

    #Load mapping file
    data = loadDataFile(MappingFile) 

    #List of which minigroups are assigned to each bundle 
    configuration = np.hstack(np.load(allocation,allow_pickle=True))

    #Get minigroups
    minigroups,minigroups_swap = getMinilpGBTGroups(data, minigroup_type)
    
    #Bundle together minigroup configuration
    bundles = getBundles(minigroups_swap,configuration)

    #Open output file
    fileout = open('allocation_20200729_1.txt', 'w')
    fileout.write( '(lpGBT_number) (number_modules) (sil=0scin=1) (layer) (u/eta) (v/phi) (number_elinks)\n' )
    for b,bundle in enumerate(bundles):
        fileout.write(str(b) + "\n")
        for minigroup in bundle:

            #list lpgbts in minigroup:
            for lpgbt in minigroups_swap[minigroup]:
                fileout.write(str(lpgbt) + " ")
                
                #Get modules associated to each lpgbt:
                data_list = data[ ((data['TPGId1']==lpgbt) | (data['TPGId2']==lpgbt)) ]
                fileout.write(str(len(data_list)) + " ")
                for index, row in data_list.iterrows():
                    if ( row['density']==2 ):
                        fileout.write("1 " + str(row['layer']) + " " + str(row['u']) + " " + str(row['v']) + " " + str(row['TPGeLinkSum']) + " " )
                    else:
                        fileout.write("0 " + str(row['layer']) + " " + str(row['u']) + " " + str(row['v']) + " " + str(row['TPGeLinkSum']) + " " )
                fileout.write("\n")
                
    fileout.close()

def produce_nTCsPerModuleHists(MappingFile,allocation,CMSSW_ModuleHists,minigroup_type="minimal",correctionConfig=None):

    #Load mapping file
    data = loadDataFile(MappingFile) 

    #List of which minigroups are assigned to each bundle 
    configuration = np.hstack(np.load(allocation,allow_pickle=True))

    #Get minigroups
    minigroups,minigroups_swap = getMinilpGBTGroups(data, minigroup_type)

    #Get list of which modules are in each minigroup
    minigroups_modules = getMiniModuleGroups(data,minigroups_swap)
    
    #Bundle together minigroup configuration
    bundles = getBundles(minigroups_swap,configuration)

    #Get nTC hists per module
    module_hists = getModuleTCHists(CMSSW_ModuleHists)
    
    #Open output file
    outfile = ROOT.TFile.Open("hists_per_bundle.root","RECREATE")
    for b,bundle in enumerate(bundles):
        outfile.mkdir("bundle_" + str(b))
        outfile.cd("bundle_" + str(b)) 
        for minigroup in bundle:

            for module in minigroups_modules[minigroup]:

                module_hists[tuple(module)].Write()

        outfile.cd()

    
def check_for_missing_modules_inMappingFile(MappingFile,CMSSW_Silicon,CMSSW_Scintillator):

    #Check for modules missing in the mapping file
    
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

def check_for_missing_modules_inCMSSW(MappingFile,CMSSW_Silicon,CMSSW_Scintillator):

    #Load external data
    data = loadDataFile(MappingFile) #dataframe    
    data_tcs_passing,data_tcs_passing_scin = getTCsPassing(CMSSW_Silicon,CMSSW_Scintillator) #from CMSSW
    getHexModuleLoadInfo(data,data_tcs_passing,data_tcs_passing_scin,True)
    
    

def study_mapping(MappingFile,CMSSW_ModuleHists,algorithm="random_hill_climb",initial_state="best_so_far",random_seed=None,max_iterations=100000,output_dir=".",print_level=0, minigroup_type="minimal",correctionConfig=None, phisplitConfig=None, include_errors_in_chi2=False):

    #Load external data
    data = loadDataFile(MappingFile) #dataframe    
    try:

        #Configuration for how to divide TCs into RegionA and RegionB (traditionally phi > 60 and phi < 60)
        split = "per_roverz_bin"
        RegionA_fixvalue_min = 55
        RegionB_fixvalue_max = None
        
        if phisplitConfig != None:
            split = phisplitConfig['type']
            if 'RegionA_fixvalue_min' in phisplitConfig.keys():
                RegionA_fixvalue_min = phisplitConfig['RegionA_fixvalue_min']
            if 'RegionB_fixvalue_max' in phisplitConfig.keys():
                RegionB_fixvalue_max = phisplitConfig['RegionB_fixvalue_max']

        inclusive_hists,module_hists = getModuleHists(CMSSW_ModuleHists, split = split, phidivisionX_fixvalue_min = phidivisionX_fixvalue_min, phidivisionY_fixvalue_max = phidivisionY_fixvalue_max)

    except EnvironmentError:
        print ( "File " + CMSSW_ModuleHists + " does not exist" )
        exit()
    # Apply various corrections to r/z distributions from CMSSW

    if correctionConfig != None:
        print ( "Applying geometry corrections" )
        applyGeometryCorrections( inclusive_hists, module_hists, correctionConfig )

    #Form hists corresponding to each lpGBT from module hists
    lpgbt_hists = getlpGBTHists(data, module_hists)

    minigroups,minigroups_swap = getMinilpGBTGroups(data, minigroup_type)
    minigroup_hists = getMiniGroupHists(lpgbt_hists,minigroups_swap,return_error_squares=include_errors_in_chi2)
    minigroup_hists_root = getMiniGroupHists(lpgbt_hists,minigroups_swap,root=True)
    #Get list of which modules are in each minigroup
    minigroups_modules = getMiniModuleGroups(data,minigroups_swap)
    
    def mapping_max(state):
        global chi2_min
        global combbest

        chi2 = 0
    
        bundles = getBundles(minigroups_swap,state)
        bundled_lpgbthists = getBundledlpgbtHists(minigroup_hists,bundles)

        max_modules = getMaximumNumberOfModulesInABundle(minigroups_modules,bundles)
        print ("max modules = ", max_modules)
        chi2 = calculateChiSquared(inclusive_hists,bundled_lpgbthists)

        typicalchi2 = 600000000000
        if include_errors_in_chi2:
            typicalchi2 = 10000000
        if (chi2<chi2_min):
            chi2_min = chi2
            combbest = np.copy(state)
            if ( print_level > 0 ):
                print (algorithm," ", chi2_min, " ", chi2_min/typicalchi2)
            if ( print_level > 1 ):
                print (repr(combbest))

        return chi2

    init_state = []
    if (initial_state == "example"):
        init_state = example_minigroup_configuration
    if (initial_state[-4:] == ".npy"):
        print (initial_state)
        init_state = np.hstack(np.load(initial_state,allow_pickle=True))
        if ( len(init_state) != len(minigroups_swap) ):
            print ( "Initial state should be the same length as the number of mini groups")
            exit()
    elif (initial_state == "random"):
        np.random.seed(random_seed)
        init_state = np.arange(len(minigroups_swap))
        np.random.shuffle(init_state)

    
    fitness_cust = mlrose.CustomFitness(mapping_max)
    # Define optimization problem object
    problem_cust = mlrose.DiscreteOpt(length = len(init_state), fitness_fn = fitness_cust, maximize = False, max_val = len(minigroups_swap), minigroups = minigroups_swap)

    # Define decay schedule
    schedule = mlrose.ExpDecay()
    #schedule = mlrose.ArithDecay()

    filename = "bundles_job_"
    filenumber = ""
    if ( len(sys.argv) > 2 ):
        filenumber = str(sys.argv[2])
    else:
        filenumber = "default"
    filename+=filenumber
    
    if ( algorithm == "save_root" ):
        #Save best combination so far into a root file
        bundles = getBundles(minigroups_swap,init_state)

        bundled_hists_root = getBundledlpgbtHistsRoot(minigroup_hists_root,bundles)
        bundled_hists = getBundledlpgbtHists(minigroup_hists,bundles)

        chi2 = calculateChiSquared(inclusive_hists,bundled_hists)
        newfile = ROOT.TFile("lpgbt_10.root","RECREATE")
        np.save(output_dir + "/" + filename + "_saveroot.npy",bundles)
        for sector in bundled_hists_root:
            for key, value in sector.items():
                value.Write()
        for sector in inclusive_hists:
            sector.Scale(1./24.)
            sector.Write()
        newfile.Close()
        print ("Chi2:",chi2)
        print ("List of Bundles:")
        for b,bundle in enumerate(bundles):
            print ("" )
            print ("bundle" + str(b) )
            for minigroup in bundle:
                #print (minigroup)
                lpgbts = minigroups_swap[minigroup]
                for lpgbt in lpgbts:
                    print (str(lpgbt) + ", "  , end = '')

    elif algorithm == "random_hill_climb" or algorithm == "simulated_annealing":

        try:
            if (algorithm == "random_hill_climb"):
                best_state, best_fitness = mlrose.random_hill_climb(problem_cust, max_attempts=10000, max_iters=max_iterations, restarts=0, init_state=init_state, random_state=random_seed)
            elif (algorithm == "simulated_annealing"):
                best_state, best_fitness = mlrose.simulated_annealing(problem_cust, schedule = schedule, max_attempts = 100000, max_iters = 10000000, init_state = init_state, random_state=random_seed)
                

        except ValueError:
            print("interrupt received, stopping and saving")

        finally:
            bundles = getBundles(minigroups_swap,combbest)
            np.save(output_dir + "/" + filename + ".npy",bundles)
            file1 = open(output_dir + "/chi2_"+filenumber+".txt","a")
            file1.write( "bundles[" + filenumber + "] = " + str(chi2_min) + "\n" )
            file1.close( )

    else:
        print("Algorithm "+ algorithm + " currently not implemented" )

    
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

    #Catch possible exceptions from batch system
    signal.signal(signal.SIGINT,handler)
    signal.signal(signal.SIGUSR1,handler)
    signal.signal(signal.SIGXCPU,handler)

    ROOT.TH1.SetDefaultSumw2()
    
    if ( config['function']['study_mapping'] ):
        subconfig = config['study_mapping']
        correctionConfig = None
        phisplitConfig = None
        include_errors_in_chi2 = False
        if 'corrections' in config.keys():
            correctionConfig = config['corrections']
        if 'include_errors_in_chi2' in subconfig.keys():
            include_errors_in_chi2 = subconfig['include_errors_in_chi2']
        if 'phisplit' in subconfig.keys():
            phisplitConfig = subconfig['phisplit']
        
            
        study_mapping(subconfig['MappingFile'],subconfig['CMSSW_ModuleHists'],algorithm=subconfig['algorithm'],initial_state=subconfig['initial_state'],random_seed=subconfig['random_seed'],max_iterations=subconfig['max_iterations'],output_dir=config['output_dir'],print_level=config['print_level'],
                      minigroup_type=subconfig['minigroup_type'],correctionConfig = correctionConfig,phisplitConfig=phisplitConfig,include_errors_in_chi2=include_errors_in_chi2
            )


    if ( config['function']['check_for_missing_modules'] ):
        subconfig = config['check_for_missing_modules']
        if ( subconfig['inMappingFile'] ):
            print("Missing modules in mapping file: "+ subconfig['MappingFile'] + "\n")
            check_for_missing_modules_inMappingFile(subconfig['MappingFile'],subconfig['CMSSW_Silicon'],subconfig['CMSSW_Scintillator'])
        if ( subconfig['inCMSSW'] ):
            print("\nMissing modules in CMSSW\n")
            check_for_missing_modules_inCMSSW(subconfig['MappingFile'],subconfig['CMSSW_Silicon'],subconfig['CMSSW_Scintillator'])

    if ( config['function']['plot_lpGBTLoads'] ):
        subconfig = config['plot_lpGBTLoads']
        plot_lpGBTLoads(subconfig['MappingFile'],subconfig['CMSSW_Silicon'],subconfig['CMSSW_Scintillator'])

    if ( config['function']['plot_ModuleLoads'] ):
        subconfig = config['plot_ModuleLoads']
        plot_ModuleLoads(subconfig['MappingFile'],subconfig['CMSSW_Silicon'],subconfig['CMSSW_Scintillator'])

    if ( config['function']['plot_ModuleLoads'] ):
        subconfig = config['plot_ModuleLoads']
        plot_ModuleLoads(subconfig['MappingFile'],subconfig['CMSSW_Silicon'],subconfig['CMSSW_Scintillator'])

    if ( config['function']['produce_AllocationFile'] ):
        subconfig = config['produce_AllocationFile']
        produce_AllocationFile(subconfig['MappingFile'],subconfig['allocation'],minigroup_type=subconfig['minigroup_type'])

    if ( config['function']['produce_nTCsPerModuleHists'] ):
        subconfig = config['produce_nTCsPerModuleHists']
        produce_nTCsPerModuleHists(subconfig['MappingFile'],subconfig['allocation'],CMSSW_ModuleHists = subconfig['CMSSW_ModuleHists'],minigroup_type=subconfig['minigroup_type'],correctionConfig=None)

    
main()
