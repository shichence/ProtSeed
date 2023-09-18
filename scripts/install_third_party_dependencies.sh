#!/bin/bash

ENV_NAME=protseed

mamba env create -n $ENV_NAME -f scripts/environment.yml
source activate $ENV_NAME
pip install "scripts/dllogger"

# Install DeepMind's OpenMM patch
work_path=$(pwd)
python_path=$(which python)
cd $(dirname $(dirname $python_path))/lib/python3.9/site-packages
patch -p0 < $work_path/lib/openmm.patch
cd $work_path

# Download folding resources
wget -q -P openfold/resources --no-check-certificate \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
