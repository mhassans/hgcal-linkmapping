#!/usr/bin/env python3
import ROOT
ROOT.gROOT.SetBatch(1)
import sys
import yaml
import numpy as np
from process import loadDataFile,loadModuleTowerMappingFile,getMinilpGBTGroups,getBundles,getBundledlpgbtHistsRoot,getMiniGroupHists,getMinilpGBTGroups,getModuleHists,getlpGBTHists,getNumberOfModulesInEachBundle,getMiniModuleGroups,getMiniTowerGroups,getTowerBundles
from geometryCorrections import applyGeometryCorrections
from root_numpy import hist2array
import matplotlib.pyplot as pl

def print_numpy_plot(hists,plotname):

    inclusive_hists = np.histogram( np.empty(0), bins = 42, range = (0.076,0.58) )

    numpy_hists = []
    for hist in hists:
        numpy_hists.append( hist2array(hist) )
    
    for bundle in numpy_hists:
        pl.step(inclusive_hists[1], np.append(bundle,bundle[-1]), where = 'post' )

    pl.ylim((0,1100000))
    pl.savefig( plotname + ".png" )
    pl.clf()
    
def print_ratio_plot(inclusive,individual,ratio,plotname):
    
    c1 = ROOT.TCanvas("c1","",800,600)

    p1 = ROOT.TPad( "p1","",0.0,0.351,1.0,1.0,0,0,0)
    p2 = ROOT.TPad( "p2","",0.0,0.0,1.0,0.35,0,0,0)
    p1.SetBottomMargin(0);
    p1.SetTopMargin(0.08);
    p2.SetBottomMargin(0.33);
    p2.SetTopMargin(0.03);
    p1.Draw()
    p2.Draw();
    p1.cd()

    inclusive.SetLineColor(ROOT.kRed)
    inclusive.SetTitle(";r/z;Number of entries")
    inclusive.Draw("HIST")
    #inclusive.Draw("E1")
    ROOT.gStyle.SetOptStat(0)
    inclusive.SetMaximum(1100E3)
    inclusive.GetYaxis().SetTitleOffset(1.9);
    inclusive.GetYaxis().SetTitleFont(43);
    inclusive.GetYaxis().SetLabelFont(43);
    inclusive.GetYaxis().SetTitleSize(25);
    inclusive.GetYaxis().SetLabelSize(25);
    for hist in individual:
        hist.Draw("HISTsame")
        #hist.Draw("E1same")
    p2.cd()
    for hist in ratio:
        hist.SetTitle(";r/z;Ratio to inclusive")
        hist.Draw("HISTsame")
        #hist.Draw("E1same")
        hist.GetYaxis().SetRangeUser(-1,3);
        hist.GetYaxis().SetTitleOffset(0.5);
        hist.GetYaxis().CenterTitle();
        hist.GetXaxis().SetTitleFont(43);
        hist.GetXaxis().SetLabelFont(43);
        hist.GetXaxis().SetTitleOffset(3.5);
        hist.GetXaxis().SetTitleSize(25);
        hist.GetXaxis().SetLabelSize(25);
        #hist.GetXaxis().SetNdivisions(505);
        hist.GetYaxis().SetNdivisions(505);
        hist.GetYaxis().SetTitleFont(43);
        hist.GetYaxis().SetLabelFont(43);
        hist.GetYaxis().SetTitleSize(25);
        hist.GetYaxis().SetLabelSize(25);
        hist.GetYaxis().SetTitleOffset(2.0);

    c1.Draw()
    c1.Print( plotname + ".png" )

def main():

    useROOT = False
    useConfiguration = False
    filein = ROOT.TFile("lpgbt_10.root")
    
    inclusive_hists = []
    phidivisionX_hists = []
    phidivisionY_hists = []
    inclusive_hists_ratio = []
    inclusive_hists_ratio_to_phidivisionY = []
    phidivisionX_hists_ratio = []
    phidivisionY_hists_ratio = []

    #Load config file if exists,
    #Otherwise assume .root file input
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        try:
            with open(config_file,'r') as file:
                config = yaml.load(file,Loader=yaml.FullLoader)
        except EnvironmentError:
            print ("Please give valid config file")
            exit()
        filein_str = config['input_file']
        if filein_str.find(".root")!=-1:
            useROOT = True
        elif filein_str.find(".npy")!=-1:
            useConfiguration = True
        else:
            print ("Please give input file in .npy or .root format")
            exit()
            
    else:
        print ("Assuming .root file input")
        useROOT = True
        


    if useConfiguration:
        
        init_state = np.hstack(np.load(filein_str,allow_pickle=True))
        MappingFile = config['npy_configuration']['mappingFile']
        TowerMappingFile = config['npy_configuration']['towerMappingFile']
        CMSSW_ModuleHists = config['npy_configuration']['CMSSW_ModuleHists']
        phisplitConfig = None
        if 'phisplit' in config['npy_configuration'].keys():
            phisplitConfig = config['npy_configuration']['phisplit']

        data = loadDataFile(MappingFile) #dataframe
        towerdata = loadModuleTowerMappingFile(TowerMappingFile)
        minigroups,minigroups_swap = getMinilpGBTGroups(data)

        #Configuration for how to divide TCs into phidivisionX and phidivisionY (traditionally phi > 60 and phi < 60)
        split = "per_roverz_bin"
        phidivisionX_fixvalue_min = 55
        phidivisionY_fixvalue_max = None
        
        if phisplitConfig != None:
            split = phisplitConfig['type']
            if 'phidivisionX_fixvalue_min' in phisplitConfig.keys():
                phidivisionX_fixvalue_min = phisplitConfig['phidivisionX_fixvalue_min']
            if 'phidivisionY_fixvalue_max' in phisplitConfig.keys():
                phidivisionY_fixvalue_max = phisplitConfig['phidivisionY_fixvalue_max']

        inclusive_hists_input,module_hists = getModuleHists(CMSSW_ModuleHists, split = split, phidivisionX_fixvalue_min = phidivisionX_fixvalue_min, phidivisionY_fixvalue_max = phidivisionY_fixvalue_max)
        if 'corrections' in config.keys():
            if config['corrections'] != None:
                print ( "Applying geometry corrections" )
                applyGeometryCorrections( inclusive_hists_input, module_hists, config['corrections'] )

        lpgbt_hists = getlpGBTHists(data, module_hists)
        minigroup_hists_root = getMiniGroupHists(lpgbt_hists,minigroups_swap,root=True)
        bundles = getBundles(minigroups_swap,init_state)
        bundled_hists = getBundledlpgbtHistsRoot(minigroup_hists_root,bundles)
        minigroups_modules = getMiniModuleGroups(data,minigroups_swap)
        nmodules = getNumberOfModulesInEachBundle(minigroups_modules,bundles)

        minigroups_towers = getMiniTowerGroups(towerdata, minigroups_modules)
        bundled_towers = getTowerBundles(minigroups_towers, bundles)
            
        inclusive = inclusive_hists_input[0].Clone("inclusive_hists_input_inclusive")
        inclusive.Add( inclusive_hists_input[1] )
        phidivisionX = inclusive_hists_input[0].Clone("inclusive_hists_input_phidivisionX")
        phidivisionY = inclusive_hists_input[1]

        inclusive.Scale(1./24)
        phidivisionX.Scale(1./24)
        phidivisionY.Scale(1./24)

        for i,(hist_phidivisionX,hist_phidivisionY) in enumerate(zip(bundled_hists[0].values(),bundled_hists[1].values())):

            inclusive_hist = hist_phidivisionX.Clone("bundle_hists_input_inclusive" + str(i))

            inclusive_hists.append(inclusive_hist)
            inclusive_hists[-1].Add( hist_phidivisionY )
            inclusive_hists_ratio.append( inclusive_hists[-1].Clone("inclusive_ratio_" + str(i) ) )
            inclusive_hists_ratio[-1].Divide(inclusive)

            phidivisionX_hists.append(hist_phidivisionX)
            phidivisionX_hists_ratio.append(hist_phidivisionX.Clone("phidivisionX_ratio_" + str(i) ))
            phidivisionX_hists_ratio[-1].Divide(phidivisionX)
            
            phidivisionY_hists.append(hist_phidivisionY)
            phidivisionY_hists_ratio.append(hist_phidivisionY.Clone("phidivisionY_ratio_" + str(i) ))
            phidivisionY_hists_ratio[-1].Divide(phidivisionY)

            inclusive_hists_ratio_to_phidivisionY.append(inclusive_hists[-1].Clone("inclusive_ratio_to_phidivisionY_" + str(i) ))
            inclusive_hists_ratio_to_phidivisionY[-1].Divide(hist_phidivisionY)
            inclusive_hists_ratio_to_phidivisionY[-1].SetLineColor(1+i)

        module_hists = None
        inclusive_hists_input = None
        minigroups = None
        minigroups_swap = None
        lpgbt_hists = None
        minigroup_hists_root = None
        bundled_hists = None

    elif useROOT:
        phidivisionX = filein.Get("ROverZ_PhiDivisionX")
        phidivisionY = filein.Get("ROverZ_PhiDivisionY")
        ROOT.TH1.Add(phidivisionX,phidivisionY)

        for i in range (24):
            phidivisionX_hists.append( filein.Get("lpgbt_ROverZ_bundled_"+str(i)+"_0") )
            phidivisionY_hists.append( filein.Get("lpgbt_ROverZ_bundled_"+str(i)+"_1") )
            inclusive_hists.append( phidivisionX_hists[-1].Clone("inclusive_hists_input_phidivisionX" + str(i)) )
            
            ROOT.TH1.Add(inclusive_hists[-1],phidivisionY_hists[-1])
            
            inclusive_hists_ratio.append (  inclusive_hists[-1].Clone ("inclusive_ratio_" + str(i)  )  )
            phidivisionX_hists_ratio.append (  phidivisionX_hists[-1].Clone ("phidivisionX_ratio_" + str(i)  )  )
            phidivisionY_hists_ratio.append (  phidivisionY_hists[-1].Clone ("phidivisionY_ratio_" + str(i)  )  )
            
            
            inclusive_hists_ratio_to_phidivisionY.append (  inclusive_hists[-1].Clone ("inclusive_ratio_to_phidivisionY_" + str(i)  )  )

            inclusive_hists_ratio[-1].Divide( inclusive )            
            phidivisionX_hists_ratio[-1].Divide( phidivisionX  )
            phidivisionY_hists_ratio[-1].Divide( phidivisionY  )
            inclusive_hists_ratio_to_phidivisionY[-1].Divide( phidivisionY  )            

    print_numpy_plot( inclusive_hists, "numpy_inclusive")
    print_numpy_plot( phidivisionX_hists, "numpy_phidivisionX")
    print_numpy_plot( phidivisionY_hists, "numpy_phidivisionY")

    print_ratio_plot(inclusive,inclusive_hists,inclusive_hists_ratio,"inclusive")
    print_ratio_plot(phidivisionX,phidivisionX_hists,phidivisionX_hists_ratio,"phidivisionX")
    print_ratio_plot(phidivisionY,phidivisionY_hists,phidivisionY_hists_ratio,"phidivisionY")
    print_ratio_plot(inclusive,inclusive_hists,inclusive_hists_ratio_to_phidivisionY,"inclusive_to_phidivisionY")

main()
