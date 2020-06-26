import pandas as pd
import numpy as np

def applyGeometryCorrections( inclusive_hists, module_hists, correctionConfig ):

    if correctionConfig['nTCCorrectionFile'] != None:
        modulesToCorrect = loadSiliconNTCCorrectionFile( correctionConfig['nTCCorrectionFile'] )
        applyNTCCorrection( module_hists, modulesToCorrect )

        # After applying all corrections, recalculate the inclusive histogram
        for i in range( len(module_hists) ):
            inclusive_hist = inclusive_hists[i]
            inclusive_hist.Reset("ICESM")

            for key, module_hist in module_hists[i].items():
                inclusive_hist.Add( module_hist )

#Special numpy implementation to be used in fluctuation.py
def applyGeometryCorrectionsNumpy( module_hists, modulesToCorrect ):

    if not modulesToCorrect.empty:
        applyNTCCorrection( module_hists, modulesToCorrect, useNumpy = True )

def applyNTCCorrection( module_hists, modulesToCorrect, useNumpy = False ):

    for index,row in modulesToCorrect.iterrows():
        dictLabel = (0, row['u'], row['v'], row['layer'])
        if dictLabel in module_hists[0].keys():
            for i in range( len(module_hists) ):
                hist = module_hists[i][dictLabel]
                correction = row['nTCsRatio']
                if useNumpy:
                    hist = hist * correction
                else:
                    originalIntegral = hist.Integral()
                    hist.Scale( correction )

                    if ( originalIntegral != 0 and hist.Integral() <= 0 ) or hist.Integral() > originalIntegral * 3.6 or hist.Integral() < originalIntegral * 0.5 :
                        print ("WARNING - Weird integral of r/z distribution after applying correction")
                        print (hist.GetEntries(),hist.Integral(),correction)

                    
def loadSiliconCorrectionFile(fileName):
    column_names = ['u', 'v', 'layer']
    modules = pd.read_csv(fileName, names=column_names)
    return modules

def loadSiliconNTCCorrectionFile(fileName):
    column_names = ['u', 'v', 'layer','nTCsRatio', 'nTCs_mappingFile', 'nTCs_cmssw']
    modules = pd.read_csv(fileName, names=column_names)
    return modules
