# HGCal link mapping studies

## Installation

- If you don't have python and the relevant packages installed please run: `source install_packages.sh`.
Note the size of the installation is around 4GB.
Then each time you start a new session run: `source start_mapping_env.sh`

- If you have python and can install your own packages the relevant ones are:
    - numpy
    - matplotlib	
    - pandas
    - pyyaml
    - scikit-learn
    - ROOT
    - root_numpy

## `main.py`

Main file containing the option to run all functions:
- `plot_lpGBTLoads`, `plot_ModuleLoads`, : Processes MC event data and determines the average number of TCs, or words, per lpGBT
- `study_mapping`, :  Find the optimised way of assigning lpGBTs to bundles, where the input options are listed for each parameter

Run using the config file `config/default.yaml`

## `process.py`

Contains the helper functions required for each function in `main.py`

## `extract_data.cxx`

Prepares the input for `process.py`. Takes a CMSSW output root file as input and produces a .csv file.

## `rotate.py` and `rotate.cxx`

Python and C++ implementations of the mapping between 120 degree HGCal sectors in (u,v) coordinates.

## `fluctuation.py`

Takes as input a choice of lpgbt bundles, and bins the trigger cell data event by event
There are several plotting scripts that investigate the impact of truncation on the number of trigger cells.
Run using the config file `config/fluctuation.yaml`

## `plotbundles.py`

Various plotting functions, mainly to plot the 24 R/Z histograms for each bundle, and take the ratio to the inclusive distribution over 24.
Run using the config file `config/plotbundles.yaml`