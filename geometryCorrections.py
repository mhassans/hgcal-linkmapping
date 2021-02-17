import pandas as pd
import numpy as np
from random import random

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
                    module_hists[i][dictLabel] = hist
                else:
                    originalIntegral = hist.Integral()
                    hist.Scale( correction )

                    if ( originalIntegral != 0 and hist.Integral() <= 0 ) or hist.Integral() > originalIntegral * 3.6 or hist.Integral() < originalIntegral * 0.5 :
                        print ("WARNING - Weird integral of r/z distribution after applying correction")
                        print (hist.GetEntries(),hist.Integral(),correction)

#Duplicate TC rawdata
def applyGeometryCorrectionsTCPtRawData( module_rawdata, modulesToCorrect ):

    if modulesToCorrect.empty:
        return

    for index,row in modulesToCorrect.iterrows():
        dictLabel = (0, row['u'], row['v'], row['layer'])
        if dictLabel in module_rawdata[0].keys():
            for i in range( len(module_rawdata) ):#phi region
                rawdata = module_rawdata[i][dictLabel]
                correction = row['nTCsRatio']

                correction_integer = np.floor(correction)
                correction_remainder = correction-correction_integer
                
                #Multiply number of entries according to correction_integer
                data1 = rawdata*int(correction_integer)
                #Then multiply entry on a random basis, more likely if correction_remainder is closer to 1
                data2 = []
                for r in rawdata:
                    if ( correction_remainder > random() ):
                        data2.append(r)

                module_rawdata[i][dictLabel] = data1+data2

                    
def loadSiliconCorrectionFile(fileName):
    column_names = ['u', 'v', 'layer']
    modules = pd.read_csv(fileName, names=column_names)
    return modules

def loadSiliconNTCCorrectionFile(fileName):
    column_names = ['u', 'v', 'layer','nTCsRatio', 'nTCs_mappingFile', 'nTCs_cmssw']
    modules = pd.read_csv(fileName, names=column_names)
    return modules


