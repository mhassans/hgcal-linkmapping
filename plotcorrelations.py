#!/usr/bin/env python3
import numpy as np
from process import loadDataFile,getMinilpGBTGroups,getBundles
from itertools import combinations
import matplotlib.pyplot as plt

def main():
    MappingFile = "data/FeMappingV7.txt"
    data = loadDataFile(MappingFile) #dataframe    
    minigroups,minigroups_swap = getMinilpGBTGroups(data)

    correlation = np.zeros((len(minigroups_swap),len(minigroups_swap)))

    listofarrs = []
    # listofarrs.append(arr1)
    # listofarrs.append(arr2)

    for job in range (10):
        listofarrs.append(np.load("bundles_job_"+str(job)+".npy"))
    
    for bundles in listofarrs:
        for bundle in bundles:
            combination = combinations(bundle, 2)
            for comb in combination:
                #print(correlation(comb))
                correlation[comb[0],comb[1]] += 1


    plt.imshow(correlation,vmin=0, vmax=3, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.savefig("temp.png")
main()
