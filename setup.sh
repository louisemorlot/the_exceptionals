#!/bin/bash

# this seems necessary for the activate call to work
source /localscratch/miniforge3/etc/profile.d/mamba.sh

# Create environment name based on the exercise name
mamba create -n exceptional python=3.10 -y
conda activate exceptional
if [[ "$CONDA_DEFAULT_ENV" == "exceptional" ]]; then
    echo "Environment activated successfully"
    # Install additional requirements
    mamba install -c pytorch -c nvidia -c conda-forge --file requirements.txt -y
else
    echo "Failed to activate the environment"
fi

## Return to base environment
# mamba deactivate

