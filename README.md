# HGCal link mapping studies

## `Process.py`

Processes MC event data and determines the average number of TCs, or words, per lpGBT


## `extract_data.cxx`

Prepares the input for `process.py`. Takes a CMSSW output root file as input and produces a .csv file.

## `rotate.py` and `rotate.cxx`

Python and C++ implementations of the mapping between 120 degree HGCal sectors in (u,v) coordinates.