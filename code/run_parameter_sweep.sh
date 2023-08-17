#!/bin/bash

# to redirect: 2>&1 | tee dakota.out

# Clean up old stuff
rm -rf P* S*
echo "Cleaned up old simulation directories"

# Setup simulations
cd ..
python setup_simulations.py
cd dakota
echo "Setup simulations"

# Source DAKOTA
source "$HOME"/.bashrc
source "$ACCOUNT_PATH"/load-dakota.sh
echo "DAKOTA sourced"

# Iterate over all trees in the parameter sweep
echo "Running parameter sweep"
declare -a arr=("P15" "S51" "S16" "S63" "P14" "S50" "P11" "P06" "P31" "P17" "P36" "S58" "S30" "S31" "P07" "S48")
for tree in "${arr[@]}"; do
    echo "Running parameter sweep for tree $tree"
    cd "${tree}"
    dakota --input 01_dakota_input.in --output 08_final_results.out --error 09_error.err > 00_output.out
    cd ..
    echo "Finished parameter sweep of tree $tree"
done

echo "Finished parameter sweep"