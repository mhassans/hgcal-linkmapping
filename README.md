# HGCal link mapping studies

## Installation

If you don't have python and the relevant packages installed please run: `source install_packages.sh`
If you additionally need `ROOT` installed, then uncomment the final line in that file.
Then each time you start a new session run: `source start_mapping_env.sh`

## `main.py`

Main file containing the option to run all functions:
- `plot_lpGBTLoads`, `plot_ModuleLoads`, : Processes MC event data and determines the average number of TCs, or words, per lpGBT
- `study_mapping`, :  Find the optimised way of assigning lpGBTs to bundles 

## `process.py`

Contains the helper functions required for each function in `main.py`

## `extract_data.cxx`

Prepares the input for `process.py`. Takes a CMSSW output root file as input and produces a .csv file.

## `rotate.py` and `rotate.cxx`

Python and C++ implementations of the mapping between 120 degree HGCal sectors in (u,v) coordinates.