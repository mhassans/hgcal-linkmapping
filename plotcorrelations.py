#!/usr/bin/env python3
import numpy as np
from process import loadDataFile,getMinilpGBTGroups,getBundles
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob

def main():
    MappingFile = "data/FeMappingV7.txt"
    FileLocation = "../outputs/"
    data = loadDataFile(MappingFile) #dataframe    
    minigroups,minigroups_swap = getMinilpGBTGroups(data)

    correlation = np.zeros((len(minigroups_swap),len(minigroups_swap)))

    listofarrs = []
    # listofarrs.append(arr1)
    # listofarrs.append(arr2)

    # for job in range (60):
    #     listofarrs.append(np.load(FileLocation + "bundles_job_"+str(job)+".npy",allow_pickle=True))
    for filename in glob(FileLocation + "bundles_job_*[0-9].npy"):
        listofarrs.append(np.load(filename,allow_pickle=True))
    for b,bundles in enumerate(listofarrs):
        print ("Bundle " + str(b) + " of " + str(len(listofarrs)))
        for bundle in bundles:
            combination = combinations(bundle, 2)
            for comb in combination:
                #print(correlation(comb))
                correlation[comb[0],comb[1]] += 1
                correlation[comb[1],comb[0]] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig1 = ax1.imshow(correlation,vmin=1, vmax=np.amax(correlation), cmap='viridis_r',norm=LogNorm(vmin=1, vmax=np.amax(correlation)))
    fig2 = ax2.imshow(correlation,vmin=0, vmax=np.amax(correlation), cmap='viridis_r')
    #plt.pcolormesh(correlation,vmin=0, vmax=np.amax(correlation), cmap='gray_r')
    #plt.pcolormesh(correlation,vmin=0.1, vmax=np.amax(correlation), cmap='gray_r',norm=LogNorm(vmin=0.01, vmax=np.amax(correlation)))


    divider1 = make_axes_locatable(ax1)
    divider2 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)

    fig.colorbar( fig1,cax=cax1)
    fig.colorbar( fig2,cax=cax2)
    fig.tight_layout()
    # plt.savefig("lin.png")
    # plt.savefig("lin.pdf")

    #plt.savefig("log.png",)
    plt.savefig("both_v2.png",dpi = 2000)
#    plt.savefig("log.pdf")
main()
