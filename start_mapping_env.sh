CondaVer=3
MINICONDA_DIR=$PWD/miniconda${CondaVer}
MAPPING_SOFTWARE=$PWD
source $MINICONDA_DIR/etc/profile.d/conda.sh
conda activate mapping_env
cd $MAPPING_SOFTWARE
export PYTHONNOUSERSITE=true
export PYTHONPATH="$PWD:$PYTHONPATH"
