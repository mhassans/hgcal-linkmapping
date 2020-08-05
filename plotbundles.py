#!/usr/bin/env python3
import ROOT
import sys
import yaml
import numpy as np
from process import loadDataFile,getMinilpGBTGroups,getBundles,getBundledlpgbtHistsRoot,getMiniGroupHists,getMinilpGBTGroups,getModuleHists,getlpGBTHists
from geometryCorrections import applyGeometryCorrections
from root_numpy import hist2array
import matplotlib.pyplot as pl

def print_numpy_plot(hists,plotname):

    inclusive_hists = np.histogram( np.empty(0), bins = 42, range = (0.076,0.58) )

    numpy_hists = []
    for hist in hists:
        numpy_hists.append( hist2array(hist) )
    
    for bundle in numpy_hists:
        #pl.step((inclusive_hists[1])[:-1], bundle , width=0.012,align='edge')
        pl.step((inclusive_hists[1])[:-1], bundle )

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
    phi60_hists = []
    inclusive_hists_ratio = []
    inclusive_hists_ratio_to_phi60 = []
    phi60_hists_ratio = []

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
        CMSSW_ModuleHists = config['npy_configuration']['CMSSW_ModuleHists'] 

        data = loadDataFile(MappingFile) #dataframe
        minigroups,minigroups_swap = getMinilpGBTGroups(data)

        inclusive_hists_input,module_hists = getModuleHists(CMSSW_ModuleHists, split = "per_roverz_bin")
        #inclusive_hists_input,module_hists = getModuleHists(CMSSW_ModuleHists, split = "fixed", RegionA_fixvalue_min = 55)
        if 'corrections' in config.keys():
            if config['corrections'] != None:
                print ( "Applying geometry corrections" )
                applyGeometryCorrections( inclusive_hists_input, module_hists, config['corrections'] )

        lpgbt_hists = getlpGBTHists(data, module_hists)
        minigroup_hists_root = getMiniGroupHists(lpgbt_hists,minigroups_swap,root=True)
        bundles = getBundles(minigroups_swap,init_state)
        bundled_hists = getBundledlpgbtHistsRoot(minigroup_hists_root,bundles)

        inclusive = inclusive_hists_input[0].Clone("inclusive_hists_input0")
        inclusive.Add( inclusive_hists_input[1] )
        phi60 = inclusive_hists_input[1]
        inclusive.Scale(1./24)
        phi60.Scale(1./24)

        for i,(hist_inc,hist_phi60) in enumerate(zip(bundled_hists[0].values(),bundled_hists[1].values())):
            inclusive_hists.append(hist_inc)
            inclusive_hists[-1].Add( hist_phi60 )
            inclusive_hists_ratio.append( inclusive_hists[-1].Clone("inclusive_ratio_" + str(i) ) )
            inclusive_hists_ratio[-1].Divide(inclusive)

            phi60_hists.append(hist_phi60)
            phi60_hists_ratio.append(hist_phi60.Clone("phi60_ratio_" + str(i) ))
            phi60_hists_ratio[-1].Divide(phi60)

            inclusive_hists_ratio_to_phi60.append(inclusive_hists[-1].Clone("inclusive_ratio_to_phi60_" + str(i) ))
            inclusive_hists_ratio_to_phi60[-1].Divide(hist_phi60)
            inclusive_hists_ratio_to_phi60[-1].SetLineColor(1+i)

        module_hists = None
        inclusive_hists_input = None
        minigroups = None
        minigroups_swap = None
        lpgbt_hists = None
        minigroup_hists_root = None
        bundled_hists = None

    elif useROOT:
        phi60 = filein.Get("ROverZ_PhiLess60")
        inclusive = filein.Get("ROverZ_PhiGreater60")
        ROOT.TH1.Add(inclusive,phi60)

        for i in range (24):
            inclusive_hists.append( filein.Get("lpgbt_ROverZ_bundled_"+str(i)+"_0") )
            phi60_hists.append( filein.Get("lpgbt_ROverZ_bundled_"+str(i)+"_1") )
            ROOT.TH1.Add(inclusive_hists[-1],phi60_hists[-1])
            
            inclusive_hists_ratio.append (  inclusive_hists[-1].Clone ("inclusive_ratio_" + str(i)  )  )
            phi60_hists_ratio.append (  phi60_hists[-1].Clone ("phi60_ratio_" + str(i)  )  )
            inclusive_hists_ratio_to_phi60.append (  inclusive_hists[-1].Clone ("inclusive_ratio_to_phi60_" + str(i)  )  )

            inclusive_hists_ratio[-1].Divide( inclusive )            
            phi60_hists_ratio[-1].Divide( phi60  )
            inclusive_hists_ratio_to_phi60[-1].Divide( phi60  )            

        
    print_ratio_plot(inclusive,inclusive_hists,inclusive_hists_ratio,"inclusive")
    print_ratio_plot(phi60,phi60_hists,phi60_hists_ratio,"phi60")
    print_ratio_plot(inclusive,inclusive_hists,inclusive_hists_ratio_to_phi60,"inclusive_to_phi60")

    print_numpy_plot( inclusive_hists, "numpy_inclusive")
    print_numpy_plot( phi60_hists, "numpy_phi60")

main()
    
